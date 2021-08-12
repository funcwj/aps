#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from aps.libs import Register
from aps.asr.transformer.impl import RelMultiheadAttention, ApsTransformerEncoderLayer

StreamingTransformerEncoderLayers = Register("streaming_xfmr_encoder_layer")


class StreamingRelMultiheadAttention(RelMultiheadAttention):
    """
    Streaming version of RelMultiheadAttention
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0,
                 bias: bool = True,
                 chunk: int = 1,
                 lctx: int = 1,
                 rctx: int = 1):
        super(StreamingRelMultiheadAttention, self).__init__(embed_dim,
                                                             num_heads,
                                                             dropout=dropout,
                                                             bias=bias)
        self.lctx = lctx * chunk
        self.rctx = rctx * chunk
        self.chunk = chunk
        self.reset()

    def reset(self):
        self.init = True
        self.cache_q = th.tensor(0)
        self.cache_k = th.tensor(0)
        self.cache_v = th.tensor(0)

    def step(self, chunk: th.Tensor, key_rel_pose: th.Tensor) -> th.Tensor:
        """
        NOTE: if chunk != 1, the results cann't match with forward ones
        Args:
            chunk (Tensor): T x N x E
            key_rel_pose (Tensor): T x D
        Return:
            chunk (Tensor): C x N x E
        """
        # only use last chunk
        if not self.init:
            chunk = chunk[-self.chunk:]
        # C x N x H x D
        query, key, value = self.inp_proj(chunk, chunk, chunk)
        # T x N x H x D
        if not self.init:
            query = th.cat([self.cache_q, query], 0)
            value = th.cat([self.cache_v, value], 0)
            key = th.cat([self.cache_k, key], 0)
        # T x N x H x T
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        term_b = th.matmul(query, key_rel_pose.transpose(0, 1))
        # T x N x H x T
        logit = term_a + term_b
        # T x N x E
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=None,
                                              key_padding_mask=None)
        context = self.wrap_out(context, weight)[0]
        self.init = False
        self.cache_q = query[self.chunk:]
        self.cache_v = value[self.chunk:]
        self.cache_k = key[self.chunk:]
        return context[self.lctx:self.lctx + self.chunk]


@StreamingTransformerEncoderLayers.register("xfmr_rel")
class StreamingTransformerRelEncoderLayer(ApsTransformerEncoderLayer):
    """
    Transformer encoder layer using relative position encodings
    """

    def __init__(self,
                 att_dim: int,
                 nhead: int,
                 lctx: int = 1,
                 rctx: int = 1,
                 feedforward_dim: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False) -> None:
        self_attn = StreamingRelMultiheadAttention(att_dim,
                                                   nhead,
                                                   lctx=lctx,
                                                   rctx=rctx,
                                                   dropout=att_dropout)
        super(StreamingTransformerRelEncoderLayer,
              self).__init__(att_dim,
                             self_attn,
                             feedforward_dim=feedforward_dim,
                             dropout=ffn_dropout,
                             activation=activation,
                             pre_norm=pre_norm)

    def reset(self):
        self.self_attn.reset()
