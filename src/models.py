"""PyTorch Implementation of Mt3/T5.

* No bias

Todo
* Integrate generator into the model, do not use a separate func
* Mask

"""
# import functools
# import operator

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter

import sys
sys.path.append("..")
from common.constants import D_NOTE, D_RHYTHM


def repar(mu, stddev, sigma=1):
    """
    Reparameterization
    """
    eps = Normal(0, sigma).sample(sample_shape=stddev.size()).to(stddev.device)
    z = mu + stddev * eps  # reparameterization trick
    return z


def log_gauss_lh(z, mu, logvar):
    """
    Calculate p(z|y), the log-likelihood of z w.r.t. a Gaussian component
    """
    llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) +
                   logvar + np.log(2 * np.pi))

    if z.ndim > 2:
        llh = torch.mean(llh, axis=-2)
    llh = torch.sum(llh, dim=1)  # sum over dimensions
    return llh


def approx_qy_x(z, mu_lookup, logvar_lookup, n_component):
    """ Reference: https://github.com/yjlolo/vae-audio/blob/master/base/base_model.py
    Refer to eq.13 in the paper https://openreview.net/pdf?id=rygkk305YQ.
    Approximating q(y|x) with p(y|z), the probability of z being assigned to class y.
    q(y|x) ~= p(y|z) = p(z|y)p(y) / p(z)

    Args:
        z (_type_): latent variables sampled from approximated posterior q(z|x)
        mu_lookup (_type_): i-th row corresponds to a mean vector of p(z|y = i) which is a Gaussian
        logvar_lookup (_type_): i-th row corresponds to a logvar vector of p(z|y = i) which is a Gaussian
        n_component (_type_): number of components of the GMM prior
    """
    device = z.device

    # log-logit of q(y|x)
    logLogit_qy_x = torch.zeros(z.shape[0], n_component).to(device)
    for k in torch.arange(0, n_component):
        mu_k, logvar_k = mu_lookup(k.to(device)), logvar_lookup(k.to(device))
        logLogit_qy_x[:, k] = log_gauss_lh(
            z, mu_k, logvar_k) + np.log(1 / n_component)

    qy_x = F.softmax(logLogit_qy_x, dim=1)
    return logLogit_qy_x, qy_x


class MTConfig(object):
    """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
    vocab_size: int = 1664
    input_dim: int = 512
    seq_len: int = 512
    target_seq_len: int = 1024
    # Activation dtypes.
    embed_dim: int = 512
    num_heads: int = 6

    num_encoder_layers: int = 8
    num_decoder_layers: int = 8

    head_dim: int = 64
    mlp_dim: int = 1024
    # Activation functions are retrieved from Flax.
    # mlp_activation: gelu, linear
    dropout_rate: float = 0.1
    # If `True`, the embedding weights are used in the decoder output layer.


def make_attention_mask(targets, dim):
    # Make encoder_decoder_mask
    batch, seq_len = targets.shape
    idx = (targets > 0).unsqueeze(-1).float()
    mask = idx.broadcast_to(batch, seq_len, dim)
    mask = mask.unsqueeze(1)
    return mask


def make_decoder_mask(targets):
    """Compute the self-attention mask for a decoder. (Assume that targets is shifted one time step after a start symbol of 0)

    Decoder mask is formed by combining a causal mask, a padding mask and an
    optional packing mask. If decoder_causal_attention is passed, it makes the
    masking non-causal for positions that have value of 1.

    The "inputs" portion of the concatenated sequence can attend to other "inputs"
    tokens even for those at a later time steps. In order to control this
    behavior, `decoder_causal_attention` is necessary. This is a binary mask with
    a value of 1 indicating that the position belonged to "inputs" portion of the
    original dataset.

    Example:

      Suppose we have a dataset with two examples.

      ds = [{"inputs": [6, 7], "targets": [8]},
            {"inputs": [3, 4], "targets": [5]}]

      After the data preprocessing with packing, the two examples are packed into
      one example with the following three fields (some fields are skipped for
      simplicity).

        targets = [[6, 7, 8, 3, 4, 5, 0]]
        decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
      decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

      where each array has [batch, length] shape with batch size being 1. Then,
      this function computes the following mask.

                        mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 0, 0],
                                  [0, 0, 0, 1, 1, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0]]]]

      mask[b, 1, :, :] represents the mask for the example `b` in the batch.
      Because mask is for a self-attention layer, the mask's shape is a square of
      shape [query length, key length].

      mask[b, 1, i, j] = 1 means that the query token at position i can attend to
      the key token at position j.

    Args:
      targets: decoder output tokens. [batch, length]

    Returns:
      the combined decoder mask.
    """
    # The same mask is applied to all attention heads. So the head dimension is 1,
    # i.e., the mask will be broadcast along the heads dim.
    # [batch, 1, length, length]
    batch, seq_len = targets.shape
    device = targets.device

    targets = F.pad(targets[:, 1:], (0, 1), 'constant', 0)

    causal_idx = torch.arange(seq_len)
    causal_mask = torch.greater_equal(causal_idx.unsqueeze(-1),
                                      causal_idx.unsqueeze(-2)).float()
    causal_mask = causal_mask.to(device)

    # Padding mask.
    idx = targets > 0
    mask = torch.multiply(idx.unsqueeze(-1), idx.unsqueeze(-2))

    # Packing mask
    mask = torch.logical_and(causal_mask.broadcast_to(batch, seq_len, seq_len),
                             mask).float()
    return mask.unsqueeze(1)


def dot_product_attention(query, key, value,
                          dropout_rate=0.1,
                          bias=None,
                          deterministic=False):
    """Computes dot-product attention given query, key, and value. Defaults to float32.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
      query: queries for calculating attention with shape of `[batch, num_heads, q_length, qk_depth_per_head]`.
      key: keys for calculating attention with shape of `[batch, num_heads, kv_length, qk_depth_per_head]`.

      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)

    Returns:
      Output of shape `[batch, length, num_heads, v_depth_per_head]`.
    """
    device = query.device
    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = torch.einsum('bqhd,bkhd->bhqk', query, key)
    # attn_weights = query @ key.transpose(-2, -1)

    # attention_bias
    if bias is not None:
        attn_weights = attn_weights + bias

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Apply attention dropout.
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate

        # T5 broadcasts along the "length" dim, but unclear which one that
        # corresponds to in positional dimensions here, assuming query dim.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1

        keep = torch.bernoulli(torch.ones(dropout_shape) * keep_prob)
        keep = torch.broadcast_to(keep, attn_weights.shape).to(device)

        attn_weights = attn_weights * keep / keep_prob

    # Take the linear combination of `value`.
    return torch.einsum('bhqk,bkhd->bqhd', attn_weights, value)
    # return attn_weights


def sinusoidal(embed_dim=512, max_len=2048,
               min_scale=1.0, max_scale=10000.0,
               dtype=np.float32):
    """Creates 1D Sinusoidal Position Embedding Initializer.
    NOTE: np.exp is done in float64 in the original t5x model. Pytorch perform `exp` in float32, which results in a slightly different result. So this function is currently implemented in numpy.

    Args:
      min_scale: Minimum frequency-scale in sine grating.
      max_scale: Maximum frequency-scale in sine grating.
      dtype: The DType of the returned values.

    Returns:
      The sinusoidal initialization function.
    """

    pe = np.zeros((max_len, embed_dim), dtype=dtype)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (embed_dim // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, embed_dim // 2) * scale_factor)
    pe[:, :embed_dim // 2] = np.sin(position * div_term)
    pe[:, embed_dim // 2:2 * (embed_dim // 2)] = np.cos(position * div_term)

    return pe


class FixedEmbed:
    def __init__(self, embed_dim):
        self.pe = sinusoidal(embed_dim)

    def get_pe(self, seq_len, decode=False):
        if decode:
            return torch.tensor(self.pe[seq_len - 2]).unsqueeze(0)

        inputs = np.arange(seq_len)[None, :]
        return torch.tensor(np.take(self.pe, inputs, axis=0))

# def FixedEmbed(embed_dim, seq_len=512):
#     pe = sinusoidal(embed_dim)
#     inputs = np.arange(seq_len)[None, :]
#     return torch.tensor(np.take(pe, inputs, axis=0))


class LayerNorm(nn.Module):
    """
    T5 Layer normalization operating on the last axis of the input data.
    No subtraction of mean or bias; Different from torch.nn.LayerNorm.
    """

    def __init__(self, d_feature=128, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.scale = Parameter(torch.empty(d_feature))
        torch.nn.init.ones_(self.scale)

    def forward(self, x):
        mean2 = torch.mean(torch.square(x), axis=-1, keepdims=True)
        x = x * torch.rsqrt(mean2 + self.eps)
        return x * self.scale


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      dropout_rate: Dropout rate. Defaults to 0.1.
    """

    def __init__(self, input_dim=512, d_hidden=2048, d_out=512, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.dropout_rate = dropout_rate

        self.fc_0 = nn.Linear(input_dim, d_hidden, bias=False)
        self.fc_1 = nn.Linear(input_dim, d_hidden, bias=False)
        self.fc_out = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x, is_train=True):
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.

        x0 = self.fc_0(x)
        x0 = F.gelu(x0)

        x1 = self.fc_1(x)

        # Take elementwise product of above intermediate activations.
        # x = functools.reduce(operator.mul, [x0, x1]) equivalent to
        x = x0 * x1

        x = F.dropout(x, p=self.dropout_rate, training=is_train)

        x = self.fc_out(x)
        return x


class MultiHeadDotProductAttention(nn.Module):
    """Multi-head dot-product attention.

      Attributes:
        num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
          should be divisible by the number of heads.
        head_dim: dimension of each head.
        dropout_rate: dropout rate.
    """

    def __init__(self, dim=512, num_heads=6, head_dim=64, self_attn=True, dropout_rate=.1):
        super(MultiHeadDotProductAttention, self).__init__()

        # assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.head_dim = head_dim
        # self.head_dim = dim // num_heads

        self.self_attn = self_attn

        # attention/query, attention/key, attention/value
        if self_attn:
            # w = concat(q_w.T, k_w.T, v_w.T)
            self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=False)
        else:
            self.qkv = nn.ModuleList(
                [nn.Linear(dim, head_dim * num_heads, bias=False) for _ in range(3)])

        # NOTE: T5 does not explicitly rescale the attention logits by 1/sqrt(depth_kq)!
        # This is folded into the initializers of the linear transformations,
        # which is equivalent under Adafactor.

        # Back to the original inputs dimensions.
        # attention/out
        # w.T
        self.fc_out = nn.Linear(head_dim * num_heads, dim, bias=False)

    def forward(self, inputs_q, inputs_kv=None, mask=None, is_train=True, decode=False):
        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim] (head_dim = dim/num_heads)
        B, N, _ = inputs_q.shape
        if self.self_attn:
            # .permute(2, 0, 3, 1, 4)
            q, k, v = self.qkv(inputs_q).reshape(
                B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4).unbind(0)
        else:
            # Todo:
            q, k, v = [l(tmp).view(B, -1, self.num_heads, self.head_dim)
                       for l, tmp in zip(self.qkv, (inputs_q, inputs_kv, inputs_kv))]
            # q, k, v = [l(tmp).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            #            for l, tmp in zip(self.qkv, (inputs_q, inputs_kv, inputs_kv))]
            # q, k, v = self.qkv(inputs_q, inputs_kv)

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            # Todo: Double check this part!!!
            attention_bias = torch.zeros_like(mask)
            attention_bias[mask <= 0] = -1e10
            # attention_bias = lax.select(
            #     mask > 0,
            #     jnp.full(mask.shape, 0.).astype(self.dtype),
            #     jnp.full(mask.shape, -1e10).astype(self.dtype))
        else:
            attention_bias = None

        # Apply attention.
        x = dot_product_attention(q, k, v,
                                  bias=attention_bias,
                                  deterministic=not is_train,
                                  dropout_rate=self.dropout_rate)

        # x = dot_product_attention(q, k,
        #                           bias=attention_bias,
        #                           deterministic=not is_train,
        #                           dropout_rate=self.dropout_rate)
        # x = (x @ v).transpose(1, 2).reshape(B, N, C)
        out = self.fc_out(x.reshape(B, N, self.head_dim * self.num_heads))

        return out


class EncoderLayer(nn.Module):

    def __init__(self, embed_dim=512, num_heads=8, mlp_dim=1024, dropout_rate=0.1):
        super().__init__()

        self.dropout_rate = dropout_rate

        # pre_attention_layer_norm/scale: shape: (512,)
        self.self_layer_norm = LayerNorm(embed_dim)  # Trainable

        # attention/key, out, query, value
        # dim=512, num_heads=6, self_attn=True, dropout_rate=.1
        self.self_attn = MultiHeadDotProductAttention(dim=embed_dim,
                                                      num_heads=num_heads,
                                                      self_attn=True,
                                                      dropout_rate=dropout_rate)

        # pre_mlp_layer_norm/scale
        self.mlp_layer_norm = LayerNorm(embed_dim)

        # mlp/wi_0, wi_1, wo
        self.mlp_block = MlpBlock(embed_dim, mlp_dim, embed_dim, dropout_rate)

    def forward(self, inputs, encoder_mask=None, is_train=True):
        x = self.self_layer_norm(inputs)

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        # Self Attn
        x = self.self_attn(inputs_q=x, mask=encoder_mask, is_train=is_train)
        x = F.dropout(x, p=self.dropout_rate, training=is_train)
        x = x + inputs

        # MLP block.
        y = self.mlp_layer_norm(x)
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = self.mlp_block(y, is_train)
        y = F.dropout(y, p=self.dropout_rate, training=is_train)
        y = y + x
        return y


class Encoder(nn.Module):
    """A stack of encoder layers."""

    def __init__(self, input_dim=512, embed_dim=512,
                 num_heads=6, mlp_dim=1024,
                 num_layers=8,
                 dropout_rate=0.1):

        super(Encoder, self).__init__()

        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        # encoder/continuous_inputs_projection
        # NOTE: potential float point issue
        self.fc = nn.Linear(input_dim, embed_dim, bias=False)

        # positional encoding
        self.fe = FixedEmbed(embed_dim)

        # encoder/layers_{0:7}
        # embed_dim=512, num_heads=8, mlp_dim=1024, dropout_rate=0.1
        self.encoder_layers = nn.ModuleList([EncoderLayer(embed_dim=embed_dim,
                                                          num_heads=num_heads,
                                                          mlp_dim=mlp_dim, dropout_rate=dropout_rate)
                                             for _ in range(num_layers)])

        # encoder/encoder_norm/scale, shape: (512,)
        self.norm = LayerNorm(embed_dim)  # trainable

    def forward(self, x, encoder_mask=None, is_train=True):

        # positional encoding, fixed/not trainable.
        seq_len = x.shape[1]
        pos_emb = self.fe.get_pe(seq_len).to(x.device)
        x = self.fc(x) + pos_emb
        x = F.dropout(x, p=self.dropout_rate, training=is_train)

        for layer in self.encoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = layer(x, encoder_mask, is_train=is_train)

        x = self.norm(x)
        x = F.dropout(x, p=self.dropout_rate, training=is_train)
        return x


class DecoderLayer(nn.Module):

    def __init__(self, embed_dim=512, num_heads=6, head_dim=64, mlp_dim=1024, dropout_rate=0.1):
        super().__init__()

        self.dropout_rate = dropout_rate

        # Self Attn
        # pre_self_attention_layer_norm/scale
        self.self_layer_norm = LayerNorm(embed_dim)
        self.head_dim = head_dim

        # attention / key, out, query, value
        self.self_attn = MultiHeadDotProductAttention(dim=embed_dim,
                                                      num_heads=num_heads,
                                                      head_dim=head_dim,
                                                      self_attn=True,
                                                      dropout_rate=dropout_rate)

        # Encoder-Decoder Attn
        # pre_cross_attention_layer_norm/scale
        self.src_layer_norm = LayerNorm(embed_dim)
        # encoder_decoder_attention/key, out, query, value
        self.src_attn = MultiHeadDotProductAttention(dim=embed_dim,
                                                     head_dim=head_dim,
                                                     num_heads=num_heads,
                                                     self_attn=False,
                                                     dropout_rate=dropout_rate)

        # MLP
        # pre_mlp_layer_norm/scale
        self.mlp_layer_norm = LayerNorm(embed_dim)
        # mlp/wi_0, wi_1, wo
        self.mlp_block = MlpBlock(embed_dim, mlp_dim, embed_dim, dropout_rate)

    def forward(self, inputs, encoded, decoder_mask=None, encoder_decoder_mask=None, is_train=True):

        # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]

        # Self-attention block
        x = self.self_layer_norm(inputs)
        x = self.self_attn(x, x, decoder_mask, is_train=is_train)
        x = F.dropout(x, p=self.dropout_rate, training=is_train)
        x = x + inputs

        # Encoder-Decoder block.
        y = self.src_layer_norm(x)
        y = self.src_attn(y, encoded, encoder_decoder_mask, is_train=is_train)
        y = F.dropout(y, p=self.dropout_rate, training=is_train)
        y = y + x

        # MLP block.
        z = self.mlp_layer_norm(y)
        z = self.mlp_block(z, is_train)
        z = F.dropout(z, p=self.dropout_rate, training=is_train)
        z = z + y

        return z


class Decoder(nn.Module):

    def __init__(self, vocab_size=1664,
                 embed_dim=512, num_heads=6, head_dim=64, mlp_dim=1024,
                 num_layers=8,
                 dropout_rate=0.1):

        super().__init__()
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate

        # seq_length = decoder_input_tokens.shape[-1]

        # Target Embedding
        # decoder/token_embedder/embedding, shape: (1664, 512), no transpose
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)

        # positional encoding
        self.fe = FixedEmbed(embed_dim)

        # decoder/layers_{0:7}
        self.decoder_layers = nn.ModuleList([DecoderLayer(embed_dim=embed_dim,
                                                          num_heads=num_heads,
                                                          head_dim=head_dim,
                                                          mlp_dim=mlp_dim,
                                                          dropout_rate=dropout_rate)
                                             for _ in range(num_layers)])
        # decoder/decoder_norm/scale
        self.norm = LayerNorm(embed_dim)

        # decoder/logits_dense/kernel, shape: (512, 1664)
        self.fc = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, inputs, encoded,
                decoder_mask=None, encoder_decoder_mask=None,
                is_train=True):

        # inputs: Decoder_input_tokens
        seq_len = inputs.shape[1]

        # positional encoding
        pos_emb = self.fe.get_pe(
            seq_len, decode=not is_train).to(inputs.device)
        x = self.tgt_embed(inputs) + pos_emb
        x = F.dropout(x, p=self.dropout_rate, training=is_train)

        for layer in self.decoder_layers:
            # [batch, length, emb_dim] -> [batch, length, emb_dim]
            x = layer(x, encoded, decoder_mask, encoder_decoder_mask, is_train)

        x = self.norm(x)
        x = F.dropout(x, p=self.dropout_rate, training=is_train)
        x = self.fc(x)
        return x


class SubEncoder(nn.Module):

    def __init__(self, embed_dim=512, z_dim=256):
        super().__init__()

        self.compute_mu = nn.Linear(embed_dim, z_dim, bias=False)
        self.compute_var = nn.Linear(embed_dim, z_dim, bias=False)

    def forward(self, x):
        mu, var = self.compute_mu(x), self.compute_var(x).exp_()
        dist = Normal(mu, var)
        return dist


# class SubEncoder(nn.Module):

#     def __init__(self, input_dim=512, embed_dim=512, z_dim=256):
#         super().__init__()

#         # Todo: No embedding layer, seriously???
#         self.gru = nn.GRU(input_dim, embed_dim, bias=False,
#                           batch_first=True, bidirectional=True)

#         self.compute_mu = nn.Linear(2 * embed_dim, z_dim, bias=False)
#         self.compute_var = nn.Linear(2 * embed_dim, z_dim, bias=False)

#     def forward(self, x):

#         # hidden_r(n_direction * batch_size * n_hidden)
#         _, x = self.gru(x)

#         # flatten batch_size * (n_hidden * bidirection)
#         x = torch.flatten(x.transpose(0, 1), start_dim=1)

#         mu, var = self.compute_mu(x), self.compute_var(x).exp_()
#         dist = Normal(mu, var)
#         return dist


class SubDecoder(nn.Module):

    def __init__(self, input_dim, d_hidden, d_z):
        """_summary_

        Args:
            input_dim (_type_): _description_
            d_hidden (_type_): _description_
            d_z (_type_): _description_
        """

        # super(SubDecoder, self).__init__()
        super().__init__()

        # fc before sub-decoder
        self.fc_in = nn.Linear(d_z, d_hidden)
        self.gru = nn.GRU(d_z + input_dim, d_hidden, batch_first=True)
        self.fc_out = nn.Linear(d_hidden, input_dim)

    def forward(self, x, z):

        seq_len = x.shape[1]
        z_stack = torch.stack([z] * seq_len, dim=1)
        x_z = torch.cat([x, z_stack], dim=-1)  # Concatenate input and latent

        h = self.fc_in(z).unsqueeze(0)
        out, _ = self.gru(x_z, h)
        out = F.log_softmax(self.fc_out(out), dim=1)

        return out


# Todo: Make Config
class SubModelConfig(object):
    num_decoder_layers: int = 2


class MT_GMVAE(nn.Module):

    # Todo: fix this
    # config: MTConfig

    def __init__(self, cfg, attr_dim=[D_RHYTHM, D_NOTE], attr_seq_len=[16, 16]):

        super(MT_GMVAE, self).__init__()

        self.n_attr = len(attr_dim)
        self.z_dim = cfg.embed_dim // self.n_attr
        self.n_component = 2
        self.attr_dim = attr_dim

        # build latent mean and variance lookup
        self.mu_lookups = nn.ModuleList([self._build_mu_lookup() for
                                         _ in range(self.n_attr)])
        self.logvar_lookups = nn.ModuleList([self._build_logvar_lookup(pow_exp=-2)
                                             for _ in range(self.n_attr)])

        self.encoder = Encoder(input_dim=cfg.input_dim,
                               embed_dim=cfg.embed_dim,
                               num_heads=cfg.num_heads, mlp_dim=cfg.mlp_dim,
                               num_layers=cfg.num_encoder_layers,
                               dropout_rate=cfg.dropout_rate)

        # self.sub_encoders = nn.ModuleList([SubEncoder(cfg.embed_dim, cfg.embed_dim, self.z_dim) for _ in range(self.n_attr)])
        self.sub_encoders = nn.ModuleList([SubEncoder(cfg.embed_dim, self.z_dim)
                                           for _ in range(self.n_attr)])

        # Assume that cfg.embed_dim == self.n_attr * self.z_dim
        self.fc = nn.Linear(cfg.embed_dim, cfg.embed_dim, bias=False)

        # for GRU: very suspicious so it's deprecated!
        # self.fc = nn.Linear(cfg.embed_dim + 1, cfg.embed_dim, bias=False)

        # Todo: Configuration
        # self.sub_decoders = nn.ModuleList([SubDecoder(attr_dim[i], cfg.embed_dim, self.z_dim)
        #                                    for i in range(self.n_attr)])

        self.sub_decoders = nn.ModuleList([Decoder(vocab_size=attr_dim[i],
                                                   embed_dim=self.z_dim,
                                                   num_heads=4,
                                                   head_dim=cfg.head_dim,
                                                   mlp_dim=128,
                                                   num_layers=1,
                                                   dropout_rate=cfg.dropout_rate)
                                           for i in range(self.n_attr)])

        self.decoder = Decoder(vocab_size=cfg.vocab_size,
                               embed_dim=cfg.embed_dim, num_heads=cfg.num_heads,
                               head_dim=cfg.head_dim,
                               mlp_dim=cfg.mlp_dim,
                               num_layers=cfg.num_decoder_layers,
                               dropout_rate=cfg.dropout_rate)

    def _build_mu_lookup(self):
        """
        Follow Xavier initialization as in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        This can also be done using a GMM on the latent space trained with vanilla autoencoders,
        as in https://arxiv.org/abs/1611.05148.
        """
        mu_lookup = nn.Embedding(self.n_component, self.z_dim)
        nn.init.xavier_uniform_(mu_lookup.weight)
        mu_lookup.weight.requires_grad = True
        return mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        """
        Follow Table 7 in the paper (https://openreview.net/pdf?id=rygkk305YQ).
        """
        logvar_lookup = nn.Embedding(self.n_component, self.z_dim)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        return logvar_lookup

    def _infer_class(self, q_z, mu_lookup, logvar_lookup):
        """
        Reference: https://github.com/yjlolo/vae-audio/blob/master/base/base_model.py

        Args:
            q_z (_type_): _description_
            mu_lookup (_type_): _description_
            logvar_lookup (_type_): _description_

        Returns:
            _type_: _description_
        """
        logLogit_qy_x, qy_x = approx_qy_x(
            q_z, mu_lookup, logvar_lookup, self.n_component)
        return logLogit_qy_x, qy_x

    def encode(self, x, mask=None, is_train=True):
        """Global Encoder
        """
        return self.encoder(x, mask, is_train)

    def decode(self,
               decoder_input_tokens,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               is_train=True):

        logits = self.decoder(
            decoder_input_tokens,
            encoded,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
            is_train=is_train)
        return logits

    def predict_latent(self, x):
        attr_dist = [self.sub_encoders[i](x) for i in range(self.n_attr)]
        attr_z = [repar(attr_dist[i].mean, attr_dist[i].stddev)
                  for i in range(self.n_attr)]
        return attr_dist, attr_z

    def forward(self, inputs, targets, attr=None, is_train=True):

        # inputs: encoder_input_tokens
        # targets: decoder_target_tokens

        # Make encoder mask
        input_seq_len = inputs.shape[1]

        # encoder_mask is actually None, remove it if it takes too much space
        # encoder_mask = torch.ones(batch, dim, dim).unsqueeze(dim=1)
        src_mask = None

        # Make padding attention masks.
        # _, seq_len = targets.shape

        if not is_train:
            # Do not mask decoder attention based on targets padding at
            # decoding/inference time.

            # decoder_mask
            tgt_mask = None

            # All ones during inference
            # encoder_decoder_mask = torch.ones(batch, seq_len, dim).unsqueeze(1)
            src_tgt_mask = None
        else:
            # decoder_mask
            tgt_mask = make_decoder_mask(targets)

            # Make encoder_decoder_mask
            src_tgt_mask = make_attention_mask(targets, input_seq_len)

        # ========================== ENCODE ====================== #

        # Global Encoder
        encoded = self.encode(inputs, src_mask, is_train)

        # Sub Encoder
        # Infer latent (low-level encoder)
        attr_dist, attr_z = self.predict_latent(encoded)

        # Infer high-level gaussian component
        y_prob = [self._infer_class(attr_z[i],
                                    self.mu_lookups[i],
                                    self.logvar_lookups[i])
                  for i in range(self.n_attr)]

        y_pred = ([tmp for tmp, _ in y_prob], [tmp for _, tmp in y_prob])

        # ========================== SUB-DECODE ====================== #
        # Low-level decoder

        # Making attr decode masks
        if not is_train:
            # Do not mask decoder attention based on targets padding at
            # decoding/inference time.
            attr_masks = [None for _ in range(self.n_attr)]

            # All ones during inference
            # encoder_decoder_mask = torch.ones(batch, seq_len, dim).unsqueeze(1)
            src_attr_masks = [None for _ in range(self.n_attr)]
        else:
            attr_masks, src_attr_masks = [], []

            for i in range(self.n_attr):
                attr_masks.append(make_decoder_mask(attr[i]))
                src_attr_masks.append(make_attention_mask(attr[i],
                                                          input_seq_len))

        attr_pred = [self.sub_decoders[i](attr[i], attr_z[i],
                                          attr_masks[i], src_attr_masks[i])
                     for i in range(self.n_attr)]

        # # convert to one_hot
        # attr_oh = [F.one_hot(attr[i], num_classes=self.attr_dim[i])
        #            for i in range(self.n_attr)]

        # GRU version: deprecated!
        # attr_pred = [self.sub_decoders[i](attr_oh[i], attr_z[i])
        #              for i in range(self.n_attr)]
        # encoded = self.fc(torch.concat([encoded,
        #                                 attr_z.unsqueeze(-1)], axis=-1))

        # ========================== GLOBAL-DECODE ====================== #
        encoded = self.fc(torch.cat(attr_z, axis=-1))

        logits = self.decoder(targets, encoded,
                              tgt_mask, src_tgt_mask, is_train)

        # return attr_pred, attr_dist, attr_z, y_pred, logits
        return attr_pred, attr_dist, y_pred, logits
