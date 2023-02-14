from statistics import mean
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import kl_divergence, Normal

import sys  # Todo: Fix this
sys.path.append("..")


def adv_loss_fn(targets_pred, targets,
                attr_pred, attr,
                alpha=5):

    # Target reconstruction
    # loss_target = F.nll_loss(targets_pred.flatten(end_dim=1),
    #                          targets.flatten(), reduction='mean')

    # nll_loss equivalent to cross_entropy(log_softmax(logits), y)
    loss_target = F.cross_entropy(targets_pred.flatten(end_dim=1),
                                  targets.flatten(), reduction='mean')
    loss = alpha * loss_target  # Todo: Why 5

    loss_attr = []
    for i, prob in enumerate(attr_pred):
        # item = F.nll_loss(prob.flatten(end_dim=1),
        #   attr[i].flatten(), reduction='mean')
        item = F.cross_entropy(prob.flatten(end_dim=1),
                               attr[i].flatten(), reduction='mean')
        loss_attr.append(item.data)
        loss += item

    loss_term = torch.stack(loss_attr + [loss_target.data])
    return loss, loss_term
    # return loss, loss_target.data, torch.stack(loss_attr)


def latent_regularized_loss_fn(attr_z, attr_stats, device=torch.device("cpu")):
    """regularization loss - Pati et al. 2019
    """
    loss_attr = []
    loss = torch.tensor([.0]).to(device)
    for z, stats in zip(*(attr_z, attr_stats)):
        D_attr = stats.reshape(-1, 1) - stats
        D_z = z[:, 0].reshape(-1, 1) - z[:, 0]
        item = torch.nn.MSELoss(reduction="mean")(torch.tanh(D_z),
                                                  torch.sign(D_attr))
        loss_attr.append(item.data)
        loss += item

    return loss, torch.stack(loss_attr)


class SemiSupervisedLoss:
    def __init__(self, mu_lookups, logvar_lookups, n_component=2, device=torch.device("cpu")):
        self.mu_lookups = mu_lookups
        self.logvar_lookups = logvar_lookups
        self.clf_criterion = torch.nn.CrossEntropyLoss()
        self.n_component = n_component
        self.device = device

    def z_loss(self, attr_dist, qy_x):
        loss_attr = []
        loss = torch.tensor([.0]).to(self.device)

        # Latent KLD
        for i, dist in enumerate(attr_dist):
            item = torch.Tensor([.0]).to(self.device)

            # number of components
            for j in torch.arange(0, self.n_component):

                # infer current p(z|y)
                mu_pz_y = self.mu_lookups[i](j.to(self.device))
                var_pz_y = self.logvar_lookups[i](j.to(self.device)).exp_()

                dist_pz_y = Normal(mu_pz_y, var_pz_y)

                avg_dist = Normal(dist.mean.mean(dim=1),
                                  dist.stddev.mean(dim=1))
                item_j = torch.mean(kl_divergence(avg_dist, dist_pz_y),
                                    dim=-1) * qy_x[i][:, j]

                item += item_j.mean()

            loss_attr.append(item.data)
            loss += item

        return loss, torch.stack(loss_attr)

    def clf_loss(self, qy_x, logLogit_qy_x):

        loss_attr = []
        loss = torch.tensor([.0]).to(self.device)

        # Classification loss -> KLD[q(y|x) || p(y)] = H(q(y|x)) - log p(y)
        for i, logLogit in enumerate(logLogit_qy_x):

            h_qy_x = torch.mean(qy_x[i] * F.log_softmax(logLogit, dim=1),
                                dim=1)
            item = (h_qy_x - np.log(1 / self.n_component)).mean()
            loss_attr.append(item)
            loss += item

        return loss, torch.stack(loss_attr)

    def __call__(self,
                 targets_pred, targets,
                 attr_pred, attr_dist, attr,
                 qy_x, logLogit_qy_x,
                 #  attr_z, attr_stats,
                 alpha=5,
                 beta=.1):

        # Semi-supervised Loss: E[log p(x|z)] - sum{l} q(y_l|X) * KL[q(z|x) || p(z|y_l)] - KL[q(y|x) || p(y)]

        # Adversarial (Reconstruction) loss
        # adv_loss, adv_loss_target, adv_loss_attr = adv_loss_fn(targets_pred, targets,
        #                                                                attr_pred, attr,
        #                                                                alpha)
        adv_loss, adv_loss_attr = adv_loss_fn(targets_pred, targets,
                                              attr_pred, attr,
                                              alpha)

        # latent KLD
        z_kld, z_kld_attr = self.z_loss(attr_dist, qy_x)

        # classification loss
        clf_loss, clf_loss_attr = self.clf_loss(qy_x, logLogit_qy_x)

        # latent regularized loss

        # latent_loss, latent_loss_attr = latent_regularized_loss_fn(
        #     attr_z, attr_stats, self.device)
        loss = adv_loss + beta * (z_kld + clf_loss)
        # loss = adv_loss + beta * (z_kld + clf_loss) + latent_loss

        # pack output
        # loss_term = {"target": torch.tensor([adv_loss_target]),
        #              "adv": adv_loss_attr,
        #              "z": z_kld_attr,
        #              "clf": clf_loss_attr,
        #              "latent": latent_loss_attr}

        # For display purpose, detached from device
        loss_term = {"loss": loss.item(),
                     "adv": adv_loss_attr.tolist(),
                     "z": z_kld_attr.tolist(),
                     "clf": clf_loss_attr.tolist(), }
        #  "latent": latent_loss_attr.tolist()}

        return loss, loss_term


class SupervisedLoss:
    "A simple loss compute and train function."

    def __init__(self, mu_lookups, logvar_lookups, device=torch.device("cpu")):
        self.mu_lookups = mu_lookups
        self.logvar_lookups = logvar_lookups
        self.clf_criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def z_loss(self, attr_dist, y):
        loss_attr = []
        loss = torch.tensor([.0]).to(self.device)

        for i, dist in enumerate(attr_dist):
            mu_pz_y = self.mu_lookups[i](y)
            var_pz_y = self.logvar_lookups[i](y).exp_()
            dist_pz_y = Normal(mu_pz_y, var_pz_y)

            # Todo: ???
            avg_dist = Normal(dist.mean.mean(dim=1), dist.stddev.mean(dim=1))
            item = torch.mean(kl_divergence(avg_dist, dist_pz_y),
                              dim=-1)

            loss_attr.append(item.mean().data)
            loss += item.mean()

        return loss, torch.stack(loss_attr)

    def clf_loss(self, qy_x, y):
        loss_attr = []
        loss = torch.tensor([.0]).to(self.device)

        # Classification loss
        for q in qy_x:
            item = self.clf_criterion(q, y)
            loss += item
            loss_attr.append(item.data)

        return loss, torch.stack(loss_attr)

    def __call__(self,
                 targets_pred, targets,
                 attr_pred, attr_dist, attr,
                 qy_x, y,
                 #  attr_z, attr_stats,
                 alpha=5, beta=.1):

        # Supervised: E[log p(x|z)] - KL[q(z|x) || p(z|y)]

        # Reconstruction loss
        # adv_loss, adv_loss_target, adv_loss_attr = adv_loss_fn(targets_pred, targets,
        #                                                                attr_pred, attr,
        #                                                                alpha)
        adv_loss, adv_loss_attr = adv_loss_fn(targets_pred, targets,
                                              attr_pred, attr,
                                              alpha)

        # latent KLD
        z_kld, z_kld_attr = self.z_loss(attr_dist, y)

        # classification loss
        clf_loss, clf_loss_attr = self.clf_loss(qy_x, y)

        # latent regularized loss
        # latent_loss, latent_loss_attr = latent_regularized_loss_fn(
        #     attr_z, attr_stats, self.device)
        loss = adv_loss + beta * (z_kld + clf_loss)
        # loss = adv_loss + beta * (z_kld + clf_loss) + latent_loss

        # pack output
        # loss_term = {"target": torch.tensor([adv_loss_target]),
        #              "adv": adv_loss_attr,
        #              "z": z_kld_attr,
        #              "clf": clf_loss_attr,
        #              "latent": latent_loss_attr}

        # For display purpose, detached from device
        loss_term = {"loss": loss.item(),
                     "adv": adv_loss_attr.tolist(),
                     "z": z_kld_attr.tolist(),
                     "clf": clf_loss_attr.tolist()}
        #  "latent": latent_loss_attr.tolist()}

        return loss, loss_term
