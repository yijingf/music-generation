import torch
import torch.nn.functional as F

def greedy_decode(encoded, decode_fn, max_len=16, start_symbol=0):
    """For decoding attributes.

    Args:
        encoded (torch.Tensor): z_attr.
        max_len (int): Maximum length of attributes.
        start_symbol (int, optional): The initial of decoded performance token sequences. 
        src_mask (torch.Tensor, optional): _description_. Defaults to None.

    Returns:
        torch.Tensor: _description_
    """
    device = encoded.device
    batch_size = encoded.shape[0]

    # batch size * 1
    tgts = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)
    probs = []

    with torch.no_grad():
        for _ in range(max_len):
            prob = decode_fn(tgts, encoded,
                             decoder_mask=None,
                             encoder_decoder_mask=None,
                             is_train=False)
            next_tokens = torch.argmax(prob[:, -1, :], dim=-1).unsqueeze(-1)
            probs.append(prob[:, -1, :].unsqueeze(0))
            tgts = torch.cat([tgts, next_tokens], dim=-1)

    probs = torch.cat(probs).transpose(1, 0)
    probs = torch.nn.functional.softmax(probs, dim=-1)
    return tgts[:, 1:], probs


def beam_search(encoded, decode_fn, max_len=512, k=4, start_symbol=0):
    # def beam_search(model, inputs, max_len=512, k=4, start_symbol=0):
    """
    k: beam_width
    """
    device = encoded.device
    batch_size = encoded.shape[0]

    # Initialize target seq: batch size * 1
    tgts = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)

    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch
        # size of the predict method.

        # Todo: Make mask here
        # encoded = model.encode(inputs, mask=None, is_train=False)
        next_probs = decode_fn(tgts, encoded, is_train=False)[:, -1, :]
        # next_probs = model(inputs, tgts, attr, is_train=False)[:, -1, :]
        vocab_size = next_probs.shape[-1]
        probs, idx = next_probs.squeeze().log_softmax(-1).topk(k=k, axis=-1)

        # repeat inputs, target
        # inputs = inputs.repeat((k, 1, 1))
        encoded = encoded.repeat((k, 1, 1))
        # attr = attr.repeat((k, 1, 1))
        tgts = tgts.repeat((k, 1, 1)).transpose(0, 1).flatten(end_dim=-2)

        next_tokens = idx.reshape(-1, 1)
        tgts = torch.cat((tgts, next_tokens), axis=-1)

        # This has to be minus one because we already produced a round
        # of predictions before the for loop.

        # Iterate over the sequence
        for i in range(max_len - 1):

            # ds = torch.utils.data.TensorDataset(inputs, tgts)
            ds = torch.utils.data.TensorDataset(encoded, tgts)
            dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

            next_probs = []

            for input, tgt in dl:

                # Todo: replace with one forward model
                logits = decode_fn(tgt, input, is_train=False)
                # logits = model(input, tgt, attr, is_train=False)
                next_probs.append(logits[:, -1, :].log_softmax(-1))

            next_probs = torch.cat(next_probs, axis=0)
            next_probs = next_probs.reshape((-1, k, next_probs.shape[-1]))

            probs = probs.unsqueeze(-1) + next_probs
            probs = probs.flatten(start_dim=1)
            probs, idx = probs.topk(k=k, axis=-1)

            next_tokens = torch.remainder(
                idx, vocab_size).flatten().unsqueeze(-1)
            best_candidates = (idx / vocab_size).long()
            best_candidates += torch.arange(batch_size,
                                            device=encoded.device).unsqueeze(-1) * k

            tgts = tgts[best_candidates].flatten(end_dim=-2)
            tgts = torch.cat((tgts, next_tokens), axis=1)

        # return the optimal, idx returned by torch.topk is in the decreasing order of prob
        tgts = tgts.reshape(-1, k, tgts.shape[-1])
        probs = F.softmax(probs, dim=-1)[:,0]
        return tgts[:, 0, 1:], probs
        
