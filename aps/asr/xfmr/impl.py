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

from torch.nn import TransformerEncoderLayer
from typing import Optional, Tuple
from aps.libs import Register

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


class ApsMultiheadAttention(nn.Module):
    """
    My own MultiheadAttention and make sure it's same as torch.nn.MultiheadAttention
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 bias: bool = True) -> None:
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
            key_rel_pose (Tensor): L x S x D
        Return:
            logit (Tensor): L x N x H x S
        """
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        # term_b = th.einsum(
        #     "...hd,...sd->...hs", query,
        #     th.repeat_interleave(key_rel_pose[:, None], query.shape[1], dim=1))
        term_b = th.matmul(query, key_rel_pose[:, None].transpose(-1, -2))
        return term_a + term_b

    def forward(self,
                query: th.Tensor,
                key: th.Tensor,
                value: th.Tensor,
                key_rel_pose: th.Tensor,
                value_rel_pose: Optional[th.Tensor],
                key_padding_mask: Optional[th.Tensor] = None,
                attn_mask: Optional[th.Tensor] = None) -> MHSAReturnType:
        """
        Args:
            query (Tensor): L x N x E
            key (Tensor): S x N x E
            value (Tensor): S x N x E
            key_rel_pose (Tensor): L x S x D
            value_rel_pose (Tensor): L x S x D
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
        if value_rel_pose is not None:
            context += th.matmul(weight, value_rel_pose[:, None])
        return self.wrap_out(context, weight)


class XlMultiheadAttention(ApsMultiheadAttention):
    """
    MultiheadAttention with relative position embedding described in:
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
    Reference code from "RelPartialLearnableMultiHeadAttn" in
        https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py#L212
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
            self.rel_u = nn.Parameter(th.Tensor(self.num_heads, self.head_dim))
            self.rel_v = nn.Parameter(th.Tensor(self.num_heads, self.head_dim))
            nn.init.normal_(self.rel_u, std=0.02)
            nn.init.normal_(self.rel_v, std=0.02)
        else:
            self.rel_u = rel_u
            self.rel_v = rel_v
        self.rel_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def shift(self, term: th.Tensor) -> th.Tensor:
        """
        Got L x N x H x S from tensor L x N x H x 2S-1
        Args:
            term (Tensor): L x N x H x 2S(L)-1
        Return:
            term (Tensor): L x N x H x S(L)
        """
        L, N, H, X = term.shape
        if L * 2 - 1 != X:
            raise RuntimeError(
                "XlMultiheadAttention: tensor shape should be: " +
                f"L x N x H x 2L-1, but got {term.shape}")
        # L x N x H x 2L
        term_pad = tf.pad(term, (1, 0))
        # L x 2L x H x N
        term_pad = term_pad.transpose(1, -1).contiguous()
        # 2L x L x H x N
        term_pad = term_pad.view(2 * L, L, H, N)
        # L x 2L-1 x H x N
        term = term_pad[1:].view(L, 2 * L - 1, H, N)
        # L x L x H x N
        term = term[:, :L]
        # L x N x H x L
        return term.transpose(1, -1)

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
        term_bd = self.shift(term_bd)
        # L x N x H x S
        return term_ac + term_bd

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


@TransformerEncoderLayers.register("xfmr")
class TransformerTorchEncoderLayer(TransformerEncoderLayer):
    """
    Wrapper for TransformerEncoderLayer (add pre-norm)
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 pre_norm: bool = False,
                 dropout: bool = 0.1,
                 activation: str = "relu") -> None:
        super(TransformerTorchEncoderLayer,
              self).__init__(d_model,
                             nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation)
        self.pre_norm = pre_norm

    def ffn(self, src: th.Tensor) -> th.Tensor:
        """
        Get output of the feedforward network
        """
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(src)))))

    def forward(self,
                src: th.Tensor,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Support for both pre-norm & post-norm
        """
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout1(att)
        if self.pre_norm:
            src = src + self.dropout2(self.ffn(self.norm2(src)))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.ffn(src))
        return src


class ApsTransformerEncoderLayer(nn.Module):
    """
    A base class for TransformerEncoderLayer
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
        # Implementation of Feedforward model
        self.feedforward = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                         _get_activation_fn(activation),
                                         nn.Dropout(dropout),
                                         nn.Linear(dim_feedforward, d_model),
                                         nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = pre_norm

    def __setstate__(self, state: str) -> None:
        if "activation" not in state:
            state["activation"] = tf.relu
        super(ApsTransformerEncoderLayer, self).__setstate__(state)


@TransformerEncoderLayers.register("xfmr_rel")
class TransformerRelEncoderLayer(ApsTransformerEncoderLayer):
    """
    TransformerEncoderLayer using relative position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False) -> None:
        self_attn = RelMultiheadAttention(d_model, nhead, dropout=dropout)
        super(TransformerRelEncoderLayer,
              self).__init__(d_model,
                             self_attn,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation,
                             pre_norm=pre_norm)

    def forward(self,
                src: th.Tensor,
                key_rel_pose: Optional[th.Tensor] = None,
                value_rel_pose: Optional[th.Tensor] = None,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             key_rel_pose,
                             value_rel_pose,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout(att)
        if self.pre_norm:
            src = src + self.feedforward(self.norm2(src))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.feedforward(src))
        return src


@TransformerEncoderLayers.register("xfmr_xl")
class TransformerXLEncoderLayer(ApsTransformerEncoderLayer):
    """
    TransformerEncoderLayer using relative position encodings
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False,
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None) -> None:
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
                src: th.Tensor,
                sin_pose: Optional[th.Tensor] = None,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             sin_pose,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout(att)
        if self.pre_norm:
            src = src + self.feedforward(self.norm2(src))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.feedforward(src))
        return src


@TransformerEncoderLayers.register("conformer")
class ConformerEncoderLayer(nn.Module):
    """
    Conformer encoder layer proposed by Google in
        Conformer: Convolution-augmented Transformer for Speech Recognition
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 kernel_size: int = 16,
                 activation: str = "swish",
                 rel_u: Optional[nn.Parameter] = None,
                 rel_v: Optional[nn.Parameter] = None):
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = XlMultiheadAttention(d_model,
                                              nhead,
                                              dropout=dropout,
                                              rel_u=rel_u,
                                              rel_v=rel_v)
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
                      stride=1,
                      padding=kernel_size), nn.BatchNorm1d(d_model), Swish(),
            nn.Conv1d(d_model, d_model, 1), nn.Dropout(p=dropout))
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
        # T x N x F
        src = self.norm2(inp)
        # T x N x F => N x F x T
        src = th.einsum("tnf->nft", src)
        out = self.convolution(src)
        # N x F x T => T x N x F
        out = th.einsum("nft->tnf", out)
        return out + inp

    def forward(self,
                src: th.Tensor,
                sin_pose: Optional[th.Tensor] = None,
                src_mask: Optional[th.Tensor] = None,
                src_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        # src: T x N x F
        # 1) FFN
        src1 = self.feedforward1(src) * 0.5 + src
        # self-attention block
        src2 = self.norm1(src1)
        att = self.self_attn(src2,
                             src2,
                             src2,
                             sin_pose,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src1 + self.dropout(att)
        # conv
        src = self.conv(src)
        # 2) FFN
        src = self.feedforward2(src) * 0.5 + src
        # layernorm
        return self.norm3(src)


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
        output = src

        for mod in self.layers:
            output = mod(output, **kwargs)

        if self.norm is not None:
            output = self.norm(output)

        return output


def get_xfmr_encoder(name: str,
                     num_layers: int,
                     att_dim: int,
                     nhead: int,
                     dim_feedforward: int = 1024,
                     dropout: float = 0.0,
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
    if name in ["xfmr", "xfmr_rel"]:
        encoder_layer = enc_layer_cls(att_dim,
                                      nhead,
                                      dim_feedforward=dim_feedforward,
                                      dropout=dropout,
                                      pre_norm=pre_norm)
    else:
        # init for xfmr_xl
        if not untie_rel:
            rel_u = nn.Parameter(th.Tensor(nhead, att_dim // nhead))
            rel_v = nn.Parameter(th.Tensor(nhead, att_dim // nhead))
            nn.init.normal_(rel_u, std=0.02)
            nn.init.normal_(rel_v, std=0.02)
        else:
            rel_u, rel_v = None, None
        if name == "xfmr_xl":
            encoder_layer = enc_layer_cls(att_dim,
                                          nhead,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          pre_norm=pre_norm,
                                          rel_u=rel_u,
                                          rel_v=rel_v)
        else:
            if pre_norm:
                raise RuntimeError("for Conformer we disable pre_norm")
            encoder_layer = enc_layer_cls(att_dim,
                                          nhead,
                                          dim_feedforward=dim_feedforward,
                                          kernel_size=kernel_size,
                                          dropout=dropout,
                                          rel_u=rel_u,
                                          rel_v=rel_v)
    return ApsTransformerEncoder(encoder_layer, num_layers, norm=final_norm)


# ----------------------------------------------------------------------------
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


def check_self_attn(index):
    S, L, N, E = 100, 100, 8, 256
    self_attn = ApsMultiheadAttention(E, 4, dropout=0)
    self_attn.train()
    query = th.rand(L, N, E)
    if index == 0:
        key, value = query, query
    elif index == 1:
        key = th.rand(S, N, E)
        value = key
    else:
        key = th.rand(S, N, E)
        value = th.rand(S, N, E)

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
    print(f"Test ApsMultiheadAttention Pass - round: {index}")


if __name__ == "__main__":
    for i in range(3):
        check_self_attn(i)
