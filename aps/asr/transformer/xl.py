#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as tf


def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_transformer_encoder(encoder_layer, num_layers, norm=None):
    if isinstance(encoder_layer, TransformerRelEncoderLayer):
        return TransformerRelEncoder(encoder_layer, num_layers, norm=norm)
    elif isinstance(encoder_layer, TransformerXLEncoderLayer):
        return TransformerXLEncoder(encoder_layer, num_layers, norm=norm)
    raise RuntimeError(f"Unknown transformer encoder layer")


class ApsMultiheadAttention(nn.Module):
    """
    NOTE:
    My own MultiheadAttention (just want to make sure it's same as torch.nn.MultiheadAttention)
    """

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True):
        super(ApsMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.in_proj_weight = nn.Parameter(th.empty(3 * embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.in_proj_weight)
        if bias:
            self.in_proj_bias = nn.Parameter(th.empty(3 * embed_dim))
            nn.init.constant_(self.in_proj_bias, 0)
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def _src_proj(self, tensor, base):
        """
        Args:
            tensor (Tensor): T x N x E
        Return:
            tensor (Tensor): T x N x H x D
        """
        index = slice(base * self.embed_dim, (base + 1) * self.embed_dim)
        tensor = tf.linear(tensor, self.in_proj_weight[index],
                           self.in_proj_bias[index])
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.head_dim)
        return tensor

    def torch_forward(self,
                      query,
                      key,
                      value,
                      key_padding_mask=None,
                      attn_mask=None):
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            output (Tensor): L x N x E
            att_weights (Tensor): N x L x S
        """
        return tf.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            None,
            None,
            False,
            self.dropout.p,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            output (Tensor): L x N x E
            att_weights (Tensor): N x L x S
        """
        # L x N x H x D
        query = self._src_proj(query, 0)
        # S x N x H x D
        key = self._src_proj(key, 1)
        value = self._src_proj(value, 2)
        # L x N x H x S
        att_weights = th.einsum("lnhd,snhd->lnhs", query / (self.head_dim)**0.5,
                                key)
        if key_padding_mask is not None:
            att_weights = att_weights.masked_fill(
                key_padding_mask[None, :, None, :], float("-inf"))
        if attn_mask is not None:
            att_weights += attn_mask[:, None, None, :]
        # L x N x H x S
        att_weights = self.dropout(th.softmax(att_weights, dim=-1))
        # L x N x H x D
        output = th.einsum("lnhs,snhd->lnhd", att_weights, value)
        # L x N x HD
        output = output.contiguous()
        output = output.view(output.size(0), -1, self.embed_dim)
        # L x N x E
        output = self.out_proj(output)
        # L x N x S => N x L x S
        att_weights = att_weights.mean(-2).transpose(0, 1)
        return output, att_weights


class RelMultiheadAttention(ApsMultiheadAttention):
    """
    MultiheadAttention with relative position embedding described in:
        Self-Attention with Relative Position Representations
    """

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True):
        super(RelMultiheadAttention, self).__init__(embed_dim,
                                                    num_heads,
                                                    dropout=dropout,
                                                    bias=bias)

    def forward(self,
                query,
                key,
                value,
                key_pos,
                value_pos,
                key_padding_mask=None,
                attn_mask=None):
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_pos (Tensor): L x S x D
            value_pos (Tensor): L x S x D
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            output (Tensor): L x N x E
            att_weights (Tensor): N x L x S
        """
        # L x N x H x D
        query = self._src_proj(query, 0)
        # S x N x H x D
        key = self._src_proj(key, 1)
        value = self._src_proj(value, 2)
        # L x N x H x S
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        L, N, H, D = query.shape
        query = query.view(L, N * H, D)
        term_b = th.einsum("...hd,...sd->...hs", query, key_pos)
        term_b = term_b.view(L, N, H, -1)
        att_weights = (term_a + term_b) / (self.head_dim)**0.5

        if key_padding_mask is not None:
            att_weights = att_weights.masked_fill(
                key_padding_mask[None, :, None, :], float("-inf"))
        if attn_mask is not None:
            att_weights += attn_mask[:, None, None, :]
        # L x N x H x S
        att_weights = self.dropout(th.softmax(att_weights, dim=-1))
        # L x N x H x D
        output = th.einsum("lnhs,snhd->lnhd", att_weights, value)
        if value_pos is not None:
            weights = att_weights.view(L, N * H, -1)
            output_add = th.einsum("...hs,...sd->...hd", weights, value_pos)
            output_add = output_add.view(L, N, H, -1)
            output += output_add
        # L x N x HD
        output = output.contiguous()
        output = output.view(output.size(0), -1, self.embed_dim)
        # L x N x E
        output = self.out_proj(output)
        # L x N x S => N x L x S
        att_weights = att_weights.mean(-2).transpose(0, 1)
        return output, att_weights


class XlMultiheadAttention(ApsMultiheadAttention):
    """
    MultiheadAttention with relative position embedding described in:
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    Reference code from "RelPartialLearnableMultiHeadAttn" in
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L212
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0,
                 bias=True,
                 rel_u=None,
                 rel_v=None):
        super(XlMultiheadAttention, self).__init__(embed_dim,
                                                   num_heads,
                                                   dropout=dropout,
                                                   bias=bias)
        if rel_u is None or rel_v is None:
            self.rel_u = nn.Parameter(th.Tensor(self.num_heads, self.head_dim))
            self.rel_v = nn.Parameter(th.Tensor(self.num_heads, self.head_dim))
            nn.init.normal_(self.rel_u, std=0.02)
            nn.init.normal_(self.rel_v, std=0.02)
        else:
            self.rel_u = rel_u
            self.rel_v = rel_v
        self.rel_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _rel_shift(self, rel_pos):
        """
        Args:
            rel_pos (Tensor): L x N x H x S
        Return:
            rel_pos (Tensor): L x N x H x S
        """
        L, N, H, S = rel_pos.shape
        zero_pad = th.zeros((L, N, H, 1),
                            device=rel_pos.device,
                            dtype=rel_pos.dtype)
        # L x N x H x S+1
        rel_pos_pad = th.cat([rel_pos, zero_pad], dim=-1)
        # L x S+1 x N x H
        rel_pos_pad = th.einsum("lnhs->lsnh", rel_pos_pad).contiguous()
        # S+1 x L x N x H
        rel_pos_pad = rel_pos_pad.view([S + 1, L, N, H])[:1]
        # S x L x N x H
        rel_pos_pad = th.einsum("slnh->lnhs", rel_pos_pad).contiguous()
        return rel_pos_pad

    def forward(self,
                query,
                key,
                value,
                sin_pos_enc,
                key_padding_mask=None,
                attn_mask=None):
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            sin_pos_enc (Tensor): S x E
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            output (Tensor): L x N x E
            att_weights (Tensor): N x L x S
        """
        # L x N x H x D
        query = self._src_proj(query, 0)
        # S x N x H x D
        key = self._src_proj(key, 1)
        value = self._src_proj(value, 2)
        # S x E
        rel_pos = self.rel_proj(sin_pos_enc)
        # S x H x D
        rel_pos = rel_pos.view(rel_pos.size(0), self.num_heads, self.head_dim)
        # L x N x H x S
        term_ac = th.einsum("lnhd,snhd->lnhs", query + self.rel_u, key)
        # L x N x H x S
        term_bd = th.einsum("lnhd,shd->lnhs", query + self.rel_v, rel_pos)
        term_bd = self._rel_shift(term_bd)
        # L x N x H x S
        att_weights = (term_ac + term_bd) / (self.head_dim)**0.5
        if key_padding_mask is not None:
            att_weights = att_weights.masked_fill(
                key_padding_mask[None, :, None, :], float("-inf"))
        if attn_mask is not None:
            att_weights += attn_mask[:, None, None, :]
        # L x N x H x S
        att_weights = self.dropout(th.softmax(att_weights, dim=-1))
        # L x N x H x D
        output = th.einsum("lnhs,snhd->lnhd", att_weights, value)
        # L x N x HD
        output = output.contiguous()
        output = output.view(output.size(0), -1, self.embed_dim)
        # L x N x E
        output = self.out_proj(output)
        # L x N x S => N x L x S
        att_weights = att_weights.mean(-2).transpose(0, 1)
        return output, att_weights


class ApsTransformerEncoderLayer(nn.Module):
    """
    A base class for TransformerEncoderLayer
    """

    def __init__(self,
                 d_model,
                 self_attn,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pre_norm=False):
        super(ApsTransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # Implementation of Feedforward model
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                         _get_activation_fn(activation),
                                         nn.Dropout(dropout),
                                         nn.Linear(dim_feedforward, d_model),
                                         nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = tf.relu
        super(ApsTransformerEncoderLayer, self).__setstate__(state)


class TransformerRelEncoderLayer(ApsTransformerEncoderLayer):
    """
    TransformerEncoderLayer using relative position encodings
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pre_norm=False):
        self_attn = RelMultiheadAttention(d_model, nhead, dropout=dropout)
        super(TransformerRelEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation,
                             pre_norm=pre_norm)

    def forward(self,
                src,
                key_pos,
                value_pos=None,
                src_mask=None,
                src_key_padding_mask=None):
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             key_pos,
                             value_pos,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout(att)
        if self.pre_norm:
            src = src + self.feedforward(self.norm2(src))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.feedforward(src))
        return src


class TransformerXLEncoderLayer(ApsTransformerEncoderLayer):
    """
    TransformerEncoderLayer using relative position encodings
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 pre_norm=False,
                 rel_u=None,
                 rel_v=None):
        self_attn = XlMultiheadAttention(d_model,
                                         nhead,
                                         dropout=dropout,
                                         rel_u=rel_u,
                                         rel_v=rel_v)
        super(TransformerXLEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation,
                             pre_norm=pre_norm)

    def forward(self,
                src,
                sin_pos_enc,
                src_mask=None,
                src_key_padding_mask=None):
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             sin_pos_enc,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout(att)
        if self.pre_norm:
            src = src + self.feedforward(self.norm2(src))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.feedforward(src))
        return src


class ApsTransformerEncoder(nn.Module):
    """
    Wrapper for a stack of N Transformer encoder layers
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(ApsTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm


class TransformerRelEncoder(ApsTransformerEncoder):
    """
    A stack of N TransformerRelEncoderLayer layers
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerRelEncoder, self).__init__(encoder_layer,
                                                    num_layers,
                                                    norm=norm)

    def forward(self,
                src,
                key_pos,
                value_pos=None,
                mask=None,
                src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output,
                         key_pos,
                         value_pos=value_pos,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerXLEncoder(ApsTransformerEncoder):
    """
    A stack of N TransformerXLEncoderLayer layers
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerXLEncoder, self).__init__(encoder_layer,
                                                   num_layers,
                                                   norm=norm)

    def forward(self, src, sin_pos_enc, mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output,
                         sin_pos_enc,
                         src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


def padding_mask(vec, device=None):
    N = vec.nelement()
    M = vec.max().item()
    templ = th.arange(M, device=vec.device).repeat([N, 1])
    mask = (templ >= vec.unsqueeze(1))
    return mask.to(device) if device is not None else mask


def prep_sub_mask(T, device="cpu"):
    mask = (th.triu(th.ones(T, T, device=device), diagonal=1) == 1).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def check_self_attn():
    S, L, N, E = 100, 100, 8, 256
    self_attn = ApsMultiheadAttention(E, 4, dropout=0)
    self_attn.train()
    key = th.rand(S, N, E)
    value = th.rand(S, N, E)
    query = th.rand(L, N, E)

    key_len = th.randint(S // 2, S, (N,))
    key_len[0] = S
    key_padding_mask = padding_mask(key_len)
    attn_mask = prep_sub_mask(S)

    my1, my2 = self_attn(query,
                         key,
                         value,
                         key_padding_mask=key_padding_mask,
                         attn_mask=attn_mask)
    th1, th2 = self_attn.torch_forward(query,
                                       key,
                                       value,
                                       key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
    assert my1.shape == th1.shape
    assert my2.shape == th2.shape
    th.testing.assert_allclose(my2, th2)
    th.testing.assert_allclose(my1, th1)


if __name__ == "__main__":
    check_self_attn()
