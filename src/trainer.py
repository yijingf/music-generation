'''
Train a GM-VAE model.

Loss:

Loss = alpha * adv_loss_target + beta * \
    (adv_loss_rhythm + adv_loss_note) + z_kld + cls_kld

adv_loss: Adversarial loss meausres the error during reconstruction midi token sequence, rhythm and note count.
z_kld:
cls_kld:
'''

import os
import time
import torch
import numpy as np

import torch.nn.functional as F

import sys  # Todo: fix this
sys.path.append("..")

from common.utils import AverageMeter
from common.constants import N_MEASURE, N_VOCAB
# from mt_gm_vae.metric import evaluate_acc
from mt_gm_vae.vae_loss import SupervisedLoss, SemiSupervisedLoss
from mt_gm_vae.decoding import beam_search, greedy_decode

# Separator
line_size = 55


class GMVAE_PerformanceMeter(object):
    def __init__(self, attr_name=["Rhythm", "NoteCnt"]):

        self.attr_name = attr_name

        # Batch Time
        self.time = AverageMeter()

        # Loss
        self.loss = AverageMeter()  # Total loss
        self.adv_loss = AverageMeter()  # Reconstruction loss
        self.acc = AverageMeter()  # Accuracy
        self.latent_acc = AverageMeter()  # Accuracy

    def update(self, loss=None, adv=None, latent=None,
               acc=None, latent_acc=None, **args):
        self.loss.update(loss)
        self.adv_loss.update(adv)
        self.acc.update(acc)
        self.latent_acc.update(latent_acc)

    def _print(self, avg=False, print_all=False):

        if avg:
            item = "avg"
            title = "Avg Batch Performance"
        else:
            item = "val"
            title = "Batch Performance"

        side = int((line_size - len(title)) / 2)
        print("=" * side + title + "=" * side)

        print(f"{'Loss:':<16} {getattr(self.loss, item):<12.4f}")
        print(f"{'Time (sec):':<16} {getattr(self.time, item):<12.4f}\n")

        cols = [""] + self.attr_name + ["Target"]
        header = "|".join([f"{i:>12}" for i in cols])
        print(header)
        print('-' * line_size)

        rows = {"AdvLoss": getattr(self.adv_loss, item).tolist()}
        if print_all:
            rows["Acc"] = getattr(self.acc, item).tolist()
            rows["LatentAcc"] = getattr(self.latent_acc, item).tolist()

        for idx, row in rows.items():
            line = f"{idx:>12}|" + "|".join([f"{i:>12.4f}" for i in row]) + "|"
            print(line)

        print('=' * line_size)
        return

    def reset_all(self):
        for key, attr in self.__dict__.items():
            if key != "attr_name":
                attr.reset()


class NoamOpt:
    """Optim wrapper that implements adaptive learning rate.
    """

    def __init__(self, trainables, lr=0.001, model_size=512, factor=1, warmup=400):
        """
        Args:
            model_size (int): Size of embedding.
            factor (float): 
            warmup (bool): 
            optimizer (torch.optim): Optimizer.
        """
        self.optimizer = torch.optim.Adam(trainables, lr=lr,
                                          weight_decay=5e-7, betas=(0.95, 0.999), eps=1e-9)
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step_forward(self, step):
        "Update parameters and rate"
        rate = self.rate(step)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        self.optimizer.zero_grad()

    def rate(self, step):
        "Implement `lrate` above"
        lr = self.factor * (self.model_size ** (-0.5) *
                            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        return lr


class GMVAE_Trainer():

    def __init__(self, model, device, supervised=True):
        self.model = model.to(device)
        self.device = device

        self.supervised = supervised
        if supervised:
            self.loss_compute = SupervisedLoss(model.mu_lookups, model.logvar_lookups,
                                               device=device)
        else:
            self.loss_compute = SemiSupervisedLoss(model.mu_lookups, model.logvar_lookups,
                                                   model.n_component, device=device)

    def _load_params(self, fname):
        if os.path.exists(fname):
            print(f"Load model from {fname}.")
            self.model.load_state_dict(
                torch.load(fname, map_location=self.device))
            self.model.eval()
            return True
        return False

    def predict_latent(self, inputs):
        inputs = inputs.to(self.device)

        with torch.no_grad():
            encoded = self.model.encode(inputs)
            attr_dist, attr_z = self.model.predict_latent(encoded)

        return attr_dist, attr_z

    def generate(self, attr_z):
        attr_z = [i.to(self.device) for i in attr_z]

        with torch.no_grad():
            targets_pred = self.model.generate(attr_z)
            targets_pred = torch.argmax(targets_pred, axis=-1)
        return targets_pred

    def get_inputs(self, entry):
        inputs, all_attr, y, targets = entry

        inputs = inputs.to(self.device)

        # Todo: validate data type for int on Cuda

        # Count frequency of active
        rhythm = all_attr['rhythm'].to(self.device)
        # rhythm_density = torch.sum(rhythm == 1, axis=1) / N_MEASURE

        # Note density
        note = all_attr['note'].to(self.device)
        # note_density = torch.sum(note, axis=1) / N_MEASURE

        attr = [rhythm, note]
        # attr_stats = [rhythm_density, note_density]
        return inputs, attr, y, targets.to(self.device)
        # return inputs, attr, attr_stats, y, targets

    def _run_through(self, entry, beta=.1):

        inputs, attr, y, targets = self.get_inputs(entry)
        # The order of attribute term should be consistent with the order of d_attr in model
        attr_prob, attr_dist, y_prob, targets_prob = self.model(inputs,
                                                                targets,
                                                                attr)

        logLogit_qy_x, qy_x = y_prob

        # anneal beta
        if self.step < 1000:
            beta = 0
        else:
            beta = min((self.step - 1000) / 1000 * beta, beta)

        y = y.to(self.device)
        targets = targets.to(self.device)

        # calculate gmm loss
        if self.supervised:
            loss, loss_term = self.loss_compute(targets_prob, targets,
                                                attr_prob, attr_dist, attr,
                                                qy_x, y,
                                                # attr_z, attr_stats,
                                                beta=beta)
        else:
            loss, loss_term = self.loss_compute(targets_prob, targets,
                                                attr_prob, attr_dist, attr,
                                                qy_x, logLogit_qy_x,
                                                # attr_z, attr_stats,
                                                beta=beta)

        # Pack output
        attr_pred = {"rhythm": torch.argmax(attr_prob[0], dim=-1),
                     "note": torch.argmax(attr_prob[1], dim=-1)}
        y_pred = {"rhythm": torch.argmax(qy_x[0], dim=-1),
                  "note": torch.argmax(qy_x[1], dim=-1)}
        targets_pred = torch.argmax(targets_prob, dim=-1)

        # acc_score = {}
        # if return_acc:
        #     acc_score = evaluate_acc(targets_pred, targets,
        #                              attr_pred, {"rhythm": rhythm,
        #                                          "note": note},
        #                              y_pred, y,
        #                              attr_name=["rhythm", "note"],
        #                              supervised=self.supervised)
        # else:
        #     acc_score = {}

        return (attr_pred, y_pred, targets_pred), loss, loss_term

    def predict(self, inputs, max_len=320):
        """Beam search is super slow on CPU!!!
        """
        # inputs, attr, attr_stats, y, targets = self.get_inputs(entry)

        with torch.no_grad():
            encoded = self.model.encoder(inputs, is_train=False)
            _, attr_z = self.model.predict_latent(encoded)

        attr_pred_tokens, _ = [greedy_decode(attr_z[i], decode_fn, max_len=N_MEASURE)
                               for i, decode_fn in enumerate(self.model.sub_decoders)]

        # Infer high-level gaussian component
        with torch.no_grad():
            y_prob = [self.model._infer_class(attr_z[i],
                                              self.model.mu_lookups[i],
                                              self.model.logvar_lookups[i])
                      for i in range(self.model.n_attr)]

        # logLogit_qy_x = [tmp for tmp, _ in y_prob]
        qy_x = [tmp for _, tmp in y_prob]

        # Global Decoder
        targets_pred, targets_prob = beam_search(
            encoded, self.model.decoder, max_len=max_len)

        # Pack output
        attr_pred = {"rhythm": attr_pred_tokens[0],
                     "note": attr_pred_tokens[1]}
        y_pred = {"rhythm": torch.argmax(qy_x[0], dim=-1),
                  "note": torch.argmax(qy_x[1], dim=-1)}
        targets_pred = torch.argmax(targets_prob, dim=-1)

        return attr_pred, y_pred, targets_pred

    def train(self, train_dl, valid_dl, model_fname,
              postfix='resume',
              lr=0.001, n_epochs=100, print_step=100,
              start_step=0, overwrite=False,
              beta=0.1, n_early_stopping_epoch=20,
              criterion="loss", **args):

        # Model/Result paths
        model_dir = os.path.dirname(model_fname)
        prefix = os.path.basename(model_fname).split(".")[0]

        # Resume training
        loaded = self._load_params(model_fname)
        if not loaded:
            print(f"Model {model_fname} not found. Exit.")
            return

        # Save model to a new file.
        if not overwrite:
            model_fname = os.path.join(model_dir, f"{prefix}_{postfix}.pt")
            prefix = os.path.basename(model_fname).split(".")[0]
            print(f"Model will be saved as {model_fname}")
        # Model already exist but don't want to overwrite
        else:
            print(f"Model {model_fname} already exist. Exit.")
            return

        best_model_fname = os.path.join(model_dir, f"{prefix}_best.pt")

        # Set Trainables
        for i in self.model.encoder.parameters():
            i.requires_grad = False

        for i in self.model.encoder.encoder_layers[-1].mlp_block.parameters():
            i.requires_grad = True
        self.model.encoder.norm.scale.requires_grad = True

        trainables = [p for p in self.model.parameters() if p.requires_grad]

        # Optimizer
        model_opt = NoamOpt(trainables, lr=lr, model_size=self.model.encoder.embed_dim)

        # Initialial states
        self.step = start_step
        epoch = int(self.step / train_dl.batch_size)
        n_batch = len(train_dl)

        # Initialize performance tracker
        best_epoch, best_loss, best_acc = 0, np.inf, 0
        train_meter = GMVAE_PerformanceMeter(["Rhythm", "NoteCnt"])
        t_start = time.time()

        # Early stopping criterion
        n_unimproved_epoch = 0

        print(f"Start training for {n_epochs} epochs.")

        for epoch in range(epoch + 1, epoch + n_epochs + 1):

            title = f" Epoch #{epoch:<3} Start"
            side = int((line_size + 10 - len(title)) / 2)
            print('*' * side + title + '*' * side)

            epoch_t_start = time.time()

            train_meter.reset_all()
            self.model.train()

            for i, entry in enumerate(train_dl):

                self.step += 1

                t_batch_start = time.time()

                # Calculate loss
                _, loss, loss_term = self._run_through(entry, beta=beta)

                if np.isnan(loss.item()):
                    print("Training diverged...")
                    return

                # Backprop, update weights
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                # Optimizer
                model_opt.step_forward(self.step)
                # optimizer.step()
                # optimizer.zero_grad()

                # Batch time in sec
                t_batch = time.time() - t_batch_start

                # Update performance meter
                train_meter.time.update(t_batch)
                train_meter.update(**loss_term)

                # print every `print_step` steps
                if self.step % print_step == 0:
                    print(f"Step {self.step}, Epoch {epoch}: {i+1}/{n_batch}")
                    train_meter._print()
                    print(f"Learning Rate: {model_opt._rate:.5f}")

            # Epoch time in min
            t_epoch = (time.time() - epoch_t_start) / 60

            title = f" Epoch #{epoch:<3} Finished "
            side = int((line_size - len(title)) / 2)
            print('*' * side + title + '*' * side)
            print(f"Epoch Time: {t_epoch:.1f} min.")
            train_meter._print(avg=True)
            print(f"Learning Rate: {model_opt._rate:.5f}")

            # Validation
            print("Start Validation.")
            _, valid_meter = self.test(valid_dl)

            # Save model
            torch.save(self.model.state_dict(), model_fname)
            print(f"Saving model to {model_fname}.")

            # Early stopping
            # Todo: use other evaluation instead of loss
            to_update = False
            if criterion == "acc":
                if valid_meter.acc.avg[-1] >= best_acc:
                    best_acc = valid_meter.acc.avg[-1]
                    to_update = True
            else:
                if valid_meter.loss.avg <= best_loss:
                    best_loss = valid_meter.loss.avg
                    to_update = True

            if to_update:
                best_epoch = epoch
                torch.save(self.model.state_dict(), best_model_fname)
                print(f"Best epoch: #{best_epoch}.")
                print(f"Saving best model to {best_model_fname}.")
                n_unimproved_epoch = 0
            else:
                n_unimproved_epoch += 1

            if n_unimproved_epoch >= n_early_stopping_epoch and self.step >= 1000:
                print("\n")
                print("---------------Training Terminated Early---------------")
                print(f"Train Loss: {train_meter.loss.avg:.6f}")
                print(f"Validation Loss: {valid_meter.loss.avg:.6f}")
                print(
                    f"Stopped at epoch: #{epoch}, best epoch: #{best_epoch}.")
                print(
                    f"Total Training Time: {(time.time() - t_start)/60:.1f} min")
                return

        print("\n")
        print("---------------Training Finished---------------")
        print(f"Train Loss: {train_meter.loss.avg:.6f}")
        print(f"Validation Loss: {valid_meter.loss.avg:.6f}")
        print(f"Best epoch: #{best_epoch}.")
        print(f"Total Training Time: {(time.time() - t_start)/60:.1f} min")

        return

    def concat(self, output):
        # Todo:
        return output

    def test(self, dl, return_acc=False):

        self.model.eval()

        output = []
        meter = GMVAE_PerformanceMeter(["Rhythm", "NoteCnt"])

        with torch.no_grad():
            for entry in dl:

                t_start = time.time()

                _, _, loss_term = self._run_through(entry)
                # _, _, loss_term, acc_score = self._test_run_through(
                #     entry, return_acc=return_acc)

                meter.update(**loss_term)

                # Evaluation
                # if return_acc:
                #     meter.update(**acc_score)

                # Todo: concatenate output
                # output.append([attr_pred, y_pred, targets_pred])

                t_batch = time.time() - t_start

                # Update performance
                meter.time.update(t_batch)

        # output = self.concat(pred)

        meter._print(avg=True, print_all=return_acc)
        return output, meter
