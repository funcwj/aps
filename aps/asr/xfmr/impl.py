#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Implementaion of multi-head attention & transformer encoder variants
"""
import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple
from aps.libs import Register
from aps.asr.xfmr.pose import digit_shift

TransformerEncoderLayers = Register("xfmr_encoder_layer")
MHSAReturnType = Tuple[th.Tensor, Optional[th.Tensor]]


class Swish(nn.Module):
    """
    Swish activation
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return inp * th.sigmoid(inp)


def _get_activation_fn(activation: str) -> nn.Module:
    """
    Return activation function for self-attention
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return Swish()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_relative_uv(shape: Tuple[int],
                     init: str = "xavier",
                     std: float = 0.02) -> nn.Parameter:
    """
    Return rel_{u,v} used in transformer-XL's MHSA
    """
    if init not in ["xavier", "normal"]:
        raise ValueError(f"Unknown init method: {init}")
    rel_mat = nn.Parameter(th.Tensor(*shape))
    if init == "xavier":
        nn.init.xavier_uniform_(rel_mat)
    if init == "uniform":
        nn.init.normal_(rel_mat, std=std)
    return rel_mat


class ApsMultiheadAttention(nn.Module):
    """
    My own MultiheadAttention and make sure it's same as torch.nn.MultiheadAttention
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 bias: bool = True,
                 use_torch: bool = True) -> None:
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
        self.use_torch = use_torch

    def inp_proj(self, query: th.Tensor, key: th.Tensor,
                 value: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
        Return:
        Args:
            query (Tensor): T x N x H x D
            key (Tensor): S x N x H x D
            value (Tensor): S x N x H x D
        """
        if th.equal(query, key) and th.equal(value, key):
            # T x N x HD*3
            stack = tf.linear(query, self.in_proj_weight, self.in_proj_bias)
            query, key, value = th.chunk(stack, 3, dim=-1)
        else:
            query = tf.linear(query, self.in_proj_weight[:self.embed_dim],
                              self.in_proj_bias[:self.embed_dim])
            if th.equal(key, value):
                stack = tf.linear(key, self.in_proj_weight[self.embed_dim:],
                                  self.in_proj_bias[self.embed_dim:])
                key, value = th.chunk(stack, 2, dim=-1)
            else:
                base = self.embed_dim
                key = tf.linear(key,
                                self.in_proj_weight[base:base + self.embed_dim],
                                self.in_proj_bias[base:base + self.embed_dim])
                base += self.embed_dim
                value = tf.linear(
                    value, self.in_proj_weight[base:base + self.embed_dim],
                    self.in_proj_bias[base:base + self.embed_dim])
        query, key, value = [
            m.view(m.shape[0], -1, self.num_heads, self.head_dim)
            for m in [query, key, value]
        ]
        return query, key, value

    def context_weight(self,
                       logit: th.Tensor,
                       value: th.Tensor,
                       key_padding_mask: Optional[th.Tensor] = None,
                       attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Return self-attention weight and context
        Args:
            logit (Tensor): L x N x H x S
            value (Tensor): S x N x H x D
        Return:
            context (Tensor): L x N x H x D
            weight (Tensor): L x N x H x S
        """
        logit = logit / (self.head_dim)**0.5
        if key_padding_mask is not None:
            logit = logit.masked_fill(key_padding_mask[None, :, None, :],
                                      float("-inf"))
        if attn_mask is not None:
            logit += attn_mask[:, None, None, :]
        # L x N x H x S
        weight = self.dropout(th.softmax(logit, dim=-1))
        # L x N x H x D
        context = th.einsum("lnhs,snhd->lnhd", weight, value)
        return context, weight

    def dot_att(self, query: th.Tensor, key: th.Tensor) -> th.Tensor:
        """
        Compute dot attention logits
        Args:
            query (Tensor): L x N x H x D
            key (tensor): S x N x H x D
        Return:
            logit (Tensor): L x N x H x S
        """
        return th.einsum("lnhd,snhd->lnhs", query, key)

    def wrap_out(self, context: th.Tensor, weight: th.Tensor) -> MHSAReturnType:
        """
        Return context & weight tensor
        Args:
            context (Tensor): L x N x H x D
            weight (Tensor): L x N x H x S
        Return:
            context (Tensor): L x N x E
            weight (Tensor): N x L x S
        """
        # L x N x HD
        context = context.contiguous().view(context.shape[0], -1,
                                            self.embed_dim)
        # L x N x E
        context = self.out_proj(context)
        # L x N x S => N x L x S
        weight = weight.mean(-2).transpose(0, 1)
        # return
        return context, weight

    def torch_forward(self,
                      query: th.Tensor,
                      key: th.Tensor,
                      value: th.Tensor,
                      key_padding_mask: Optional[th.Tensor] = None,
                      attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            context (Tensor): L x N x E
            weight (Tensor): N x L x S
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

    def forward(self,
                query: th.Tensor,
                key: th.Tensor,
                value: th.Tensor,
                placehold: Optional[th.Tensor],
                key_padding_mask: Optional[th.Tensor] = None,
                attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            placehold (None): keep compatiable with rel/xl-attention layer
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            context (Tensor): L x N x E
            weight (Tensor): N x L x S
        """
        if self.use_torch:
            return self.torch_forward(query,
                                      key,
                                      value,
                                      key_padding_mask=key_padding_mask,
                                      attn_mask=attn_mask)
        # query: L x N x H x D
        # key, value: S x N x H x D
        query, key, value = self.inp_proj(query, key, value)
        # L x N x H x S
        logit = self.dot_att(query, key)
        # L x N x E, N x L x S
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask)
        return self.wrap_out(context, weight)


class RelMultiheadAttention(ApsMultiheadAttention):
    """
    MultiheadAttention with relative position embedding described in:
        Self-Attention with Relative Position Representations
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 bias: bool = True) -> None:
        super(RelMultiheadAttention, self).__init__(embed_dim,
                                                    num_heads,
                                                    dropout=dropout,
                                                    bias=bias)

    def dot_att(self, query: th.Tensor, key: th.Tensor,
                key_rel_pose: th.Tensor) -> th.Tensor:
        """
        Compute dot attention logits
        Args:
            query (Tensor): L x N x H x D
            key (tensor): S x N x H x D
            key_rel_pose (Tensor): 2L(S)-1 x D
        Return:
            logit (Tensor): L x N x H x S
        """
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        # 1) key_rel_pose is L x S x D
        #   a)  term_b = th.einsum(
        #           "...hd,...sd->...hs", query,
        #       th.repeat_interleave(key_rel_pose[:, None], query.shape[1], dim=1))
        #   b) term_b = th.matmul(query, key_rel_pose[:, None].transpose(-1, -2))
        # 2) key_rel_pose is 2L-1 x D
        # L x N x H x 2L-1
        term_b = th.matmul(query, key_rel_pose.transpose(0, 1))
        # L x N x H x S
        return term_a + digit_shift(term_b)

    def forward(self,
                query: th.Tensor,
                key: th.Tensor,
                value: th.Tensor,
                key_rel_pose: th.Tensor,
                key_padding_mask: Optional[th.Tensor] = None,
                attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_rel_pose (Tensor): 2L(S)-1 x D
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            context (Tensor): L x N x E
            weight (Tensor): N x L x S
        """
        # query: L x N x H x D
        # key, value: S x N x H x D
        query, key, value = self.inp_proj(query, key, value)
        # L x N x H x S
        logit = self.dot_att(query, key, key_rel_pose)
        # context: L x N x H x D
        # weight: L x N x H x S
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask)
        return self.wrap_out(context, weight)


class XlMultiheadAttention(ApsMultiheadAttention):
    """
    MultiheadAttention with relative position embedding described in:
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 bias: bool = True,
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None) -> None:
        super(XlMultiheadAttention, self).__init__(embed_dim,
                                                   num_heads,
                                                   dropout=dropout,
                                                   bias=bias)
        if rel_u is None or rel_v is None:
            self.rel_u = _get_relative_uv((self.num_heads, self.head_dim))
            self.rel_v = _get_relative_uv((self.num_heads, self.head_dim))
        else:
            self.rel_u = rel_u
            self.rel_v = rel_v
        self.rel_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def dot_att(self, query: th.Tensor, key: th.Tensor,
                sin_pose: th.Tensor) -> th.Tensor:
        """
        Compute dot attention logits
        Args:
            query (Tensor): L x N x H x D
            key (tensor): S(L) x N x H x D
            sin_pose (Tensor): 2L-1 x E
        Return:
            logit (Tensor): L x N x H x S(L)
        """
        # L x N x H x S
        term_ac = th.einsum("lnhd,snhd->lnhs", query + self.rel_u, key)
        # 2S-1 x E => 2S-1 x H x D
        rel_pos = self.rel_proj(sin_pose)
        rel_pos = rel_pos.view(-1, self.num_heads, self.head_dim)
        # L x N x H x 2S-1
        term_bd = th.einsum("lnhd,shd->lnhs", query + self.rel_v, rel_pos)
        # L x N x H x S
        return term_ac + digit_shift(term_bd)

    def forward(self,
                query: th.Tensor,
                key: th.Tensor,
                value: th.Tensor,
                sin_pose: th.Tensor,
                key_padding_mask: Optional[th.Tensor] = None,
                attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            sin_pose (Tensor): 2S-1 x E
            key_padding_mask (Tensor): N x S
            attn_mask (Tensor): L x S, additional mask
        Return:
            context (Tensor): L x N x E
            weight (Tensor): N x L x S
        """
        # query: L x N x H x D
        # key, value: S x N x H x D
        query, key, value = self.inp_proj(query, key, value)
        # L x N x H x S
        logit = self.dot_att(value, key, sin_pose)
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask)
        return self.wrap_out(context, weight)


class ApsTransformerEncoderLayer(nn.Module):
    """
    The base class for Transformer encoder layer
    """

    def __init__(self,
                 d_model: int,
                 self_attn: nn.Module,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False) -> None:
        super(ApsTransformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        # implementation of feedforward model
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                         _get_activation_fn(activation),
                                         nn.Dropout(dropout),
                                         nn.Linear(dim_feedforward, d_model),
                                         nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def forward(self,
                src: th.Tensor,
                inj_pose: Optional[th.Tensor] = None,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            src (Tensor): T x N x D
            src_mask (None or Tensor): T x T
            inj_pose (None or Tensor): injected positional encodings
            src_key_padding_mask (None or Tensor): N x T
        Return:
            out (Tensor): T x N x D
        """
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             inj_pose,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout(att)
        if self.pre_norm:
            src = src + self.feedforward(self.norm2(src))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.feedforward(src))
        return src


class ApsConformerEncoderLayer(nn.Module):
    """
    The base class for Conformer encoder layer proposed by Google in
    Conformer: Convolution-augmented Transformer for Speech Recognition
    """

    def __init__(self,
                 d_model: int,
                 self_attn: nn.Module,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 kernel_size: int = 16,
                 activation: str = "swish"):
        super(ApsConformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feedforward1 = nn.Sequential(nn.LayerNorm(d_model),
                                          nn.Linear(d_model, dim_feedforward),
                                          _get_activation_fn(activation),
                                          nn.Dropout(dropout),
                                          nn.Linear(dim_feedforward, d_model),
                                          nn.Dropout(dropout))
        self.convolution = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), nn.GLU(dim=-2),
            nn.Conv1d(d_model,
                      d_model,
                      kernel_size * 2 + 1,
                      groups=d_model,
                      padding=kernel_size), nn.BatchNorm1d(d_model),
            _get_activation_fn(activation), nn.Conv1d(d_model, d_model, 1),
            nn.Dropout(p=dropout))
        self.feedforward2 = nn.Sequential(nn.LayerNorm(d_model),
                                          nn.Linear(d_model, dim_feedforward),
                                          _get_activation_fn(activation),
                                          nn.Dropout(dropout),
                                          nn.Linear(dim_feedforward, d_model),
                                          nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def conv(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): T x N x D
        Return
            out (Tensor): T x N x D
        """
        # T x N x F
        src = self.norm2(inp)
        # T x N x F => N x F x T
        src = th.einsum("tnf->nft", src)
        out = self.convolution(src)
        # N x F x T => T x N x F
        out = th.einsum("nft->tnf", out)
        return out

    def forward(self,
                src: th.Tensor,
                inj_pose: Optional[th.Tensor] = None,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            src (Tensor): T x N x D
            inj_pose (None or Tensor): injected positional encodings
            src_mask (None or Tensor): T x T
            src_key_padding_mask (None or Tensor): N x T
        Return:
            out (Tensor): T x N x D
        """
        # 1) FFN
        src1 = self.feedforward1(src) * 0.5 + src
        # self-attention block
        src2 = self.norm1(src1)
        att = self.self_attn(src2,
                             src2,
                             src2,
                             inj_pose,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src1 + self.dropout(att)
        # conv
        src = self.conv(src) + src
        # 2) FFN
        src = self.feedforward2(src) * 0.5 + src
        # layernorm
        return self.norm3(src)


@TransformerEncoderLayers.register("xfmr_abs")
class TransformerEncoderLayer(ApsTransformerEncoderLayer):
    """
    Standard Transformer encoder layer using absolute position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 pre_norm: bool = False,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 activation: str = "relu") -> None:
        self_attn = ApsMultiheadAttention(d_model,
                                          nhead,
                                          dropout=att_dropout,
                                          use_torch=True)
        super(TransformerEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             pre_norm=pre_norm)


@TransformerEncoderLayers.register("xfmr_rel")
class TransformerRelEncoderLayer(ApsTransformerEncoderLayer):
    """
    Transformer encoder layer using relative position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False) -> None:
        self_attn = RelMultiheadAttention(d_model, nhead, dropout=att_dropout)
        super(TransformerRelEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             pre_norm=pre_norm)


@TransformerEncoderLayers.register("xfmr_xl")
class TransformerXLEncoderLayer(ApsTransformerEncoderLayer):
    """
    Transformer encoder layer using relative position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False,
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None) -> None:
        self_attn = XlMultiheadAttention(d_model,
                                         nhead,
                                         dropout=att_dropout,
                                         rel_u=rel_u,
                                         rel_v=rel_v)
        super(TransformerXLEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             pre_norm=pre_norm)


@TransformerEncoderLayers.register("cfmr_abs")
class ConformerEncoderLayer(ApsConformerEncoderLayer):
    """
    Conformer encoder layer using absolute position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 pre_norm: bool = True,
                 kernel_size: int = 16,
                 activation: str = "swish") -> None:
        self_attn = ApsMultiheadAttention(d_model,
                                          nhead,
                                          dropout=att_dropout,
                                          use_torch=True)
        super(ConformerEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             kernel_size=kernel_size)


@TransformerEncoderLayers.register("cfmr_rel")
class ConformerRelEncoderLayer(ApsConformerEncoderLayer):
    """
    Conformer encoder layer using relative position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 pre_norm: bool = True,
                 kernel_size: int = 16,
                 activation: str = "swish",
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None) -> None:
        self_attn = RelMultiheadAttention(d_model, nhead, dropout=att_dropout)
        super(ConformerRelEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             kernel_size=kernel_size)


@TransformerEncoderLayers.register("cfmr_xl")
class ConformerXLEncoderLayer(ApsConformerEncoderLayer):
    """
    Conformer encoder layer that uses relative position encoding proposed in Transformer-XL
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 pre_norm: bool = True,
                 kernel_size: int = 16,
                 activation: str = "swish",
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None) -> None:
        self_attn = XlMultiheadAttention(d_model,
                                         nhead,
                                         dropout=att_dropout,
                                         rel_u=rel_u,
                                         rel_v=rel_v)
        super(ConformerXLEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=ffn_dropout,
                             activation=activation,
                             kernel_size=kernel_size)


class ApsTransformerEncoder(nn.Module):
    """
    Wrapper for a stack of N Transformer encoder layers
    """
    __constants__ = ['norm']

    def __init__(self,
                 encoder_layer: nn.Module,
                 num_layers: int,
                 norm: Optional[nn.Module] = None) -> None:
        super(ApsTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: th.Tensor, **kwargs) -> th.Tensor:
        """
        Args:
            src (Tensor): T x N x D
        Return:
            out (Tensor): T x N x D
        """
        out = src

        for mod in self.layers:
            out = mod(out, **kwargs)

        if self.norm is not None:
            out = self.norm(out)

        return out


def get_xfmr_encoder(name: str,
                     num_layers: int,
                     att_dim: int,
                     nhead: int,
                     dim_feedforward: int = 1024,
                     att_dropout: float = 0.0,
                     ffn_dropout: float = 0.0,
                     kernel_size: int = 16,
                     pre_norm: bool = True,
                     untie_rel: bool = True) -> nn.Module:
    """
    Return transformer based encoders
    """
    if name not in TransformerEncoderLayers:
        raise ValueError(f"Unknown type of the encoders: {name}")
    final_norm = nn.LayerNorm(att_dim) if pre_norm else None
    enc_layer_cls = TransformerEncoderLayers[name]
    enc_kwargs = {
        "pre_norm": pre_norm,
        "att_dropout": att_dropout,
        "ffn_dropout": ffn_dropout,
        "dim_feedforward": dim_feedforward
    }
    arch, att = name.split("_")
    # for conformer
    if arch == "cfmr":
        enc_kwargs["kernel_size"] = kernel_size
    # for xl-attention
    if att == "xl":
        if not untie_rel:
            rel_u = _get_relative_uv((nhead, att_dim // nhead))
            rel_v = _get_relative_uv((nhead, att_dim // nhead))
            print("Tie relative trainable parameters", flush=True)
        else:
            rel_u, rel_v = None, None
        enc_kwargs["rel_u"] = rel_u
        enc_kwargs["rel_v"] = rel_v
    return ApsTransformerEncoder(enc_layer_cls(att_dim, nhead, **enc_kwargs),
                                 num_layers,
                                 norm=final_norm)
