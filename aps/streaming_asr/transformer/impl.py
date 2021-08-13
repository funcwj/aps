#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Dict
from aps.libs import Register
from aps.asr.transformer.impl import RelMultiheadAttention, ApsTransformerEncoderLayer, ApsTransformerEncoder

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
        assert lctx + rctx != 0
        # NOTE: now only supports chunk == 1
        assert chunk == 1
        self.lctx = lctx * chunk
        self.chunk = chunk
        self.cache = (lctx + rctx) * chunk
        self.reset()

    def reset(self):
        self.t = 0
        self.cache_q = th.tensor(0)
        self.cache_k = th.tensor(0)
        self.cache_v = th.tensor(0)

    def step(self, chunk: th.Tensor, key_rel_pose: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            key_rel_pose (Tensor): T x D
        Return:
            chunk (Tensor): C x N x E
        """
        if self.t:
            chunk = chunk[-self.chunk:]
        # C x N x H x D
        query, key, value = self.inp_proj(chunk, chunk, chunk)
        # T x N x H x D
        if self.t:
            query = th.cat([self.cache_q, query], 0)
            value = th.cat([self.cache_v, value], 0)
            key = th.cat([self.cache_k, key], 0)
        num_frames = query.shape[0]
        rel_pose = key_rel_pose[-num_frames:]
        # T x N x H x T
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        term_b = th.matmul(query, rel_pose.transpose(0, 1))
        # T x N x H x T
        logit = term_a + term_b
        # T x N x E
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=None,
                                              key_padding_mask=None)
        context = self.wrap_out(context, weight)[0]
        self.t = min(self.t, self.lctx) + self.chunk
        self.cache_q = query[-self.cache:]
        self.cache_v = value[-self.cache:]
        self.cache_k = key[-self.cache:]
        return context[self.t - self.chunk:self.t]


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
                                                   chunk=1,
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

    def step(self, chunk: th.Tensor, inj_pose: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            inj_pose (Tensor): T x D
        Return:
            chunk (Tensor): T x N x E
        """
        inp = chunk
        if self.pre_norm:
            chunk = self.norm1(chunk)
        att = self.self_attn.step(chunk, inj_pose)
        chunk = inp + self.dropout(att)
        if self.pre_norm:
            chunk = chunk + self.feedforward(self.norm2(chunk))
        else:
            chunk = self.norm1(chunk)
            chunk = self.norm2(chunk + self.feedforward(chunk))
        return chunk


class ApsStreamingTransformerEncoder(ApsTransformerEncoder):
    """
    Wrapper for stack of N streaming Transformer encoder layers
    """

    def __init__(self,
                 encoder_layer: nn.Module,
                 num_layers: int,
                 norm: Optional[nn.Module] = None) -> None:
        super(ApsStreamingTransformerEncoder, self).__init__(encoder_layer,
                                                             num_layers,
                                                             norm=norm)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def step(self,
             chunk: th.Tensor,
             inj_pose: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            inj_pose (Tensor): T x D
        Return:
            out (Tensor): T x N x E
        """
        out = chunk

        for mod in self.layers:
            out = mod.step(out, inj_pose=inj_pose)

        if self.norm is not None:
            out = self.norm(out)

        return out


def get_xfmr_encoder(arch: str, pose: str, num_layers: int,
                     arch_kwargs: Dict) -> nn.Module:
    """
    Return transformer based encoders
    """
    name = f"{arch}_{pose}"
    if name not in StreamingTransformerEncoderLayers:
        raise ValueError(f"Unknown type of the encoders: {name}")
    att_dim = arch_kwargs["att_dim"]
    if "pre_norm" in arch_kwargs and arch_kwargs["pre_norm"]:
        final_norm = nn.LayerNorm(att_dim)
    else:
        final_norm = None
    enc_layer_cls = StreamingTransformerEncoderLayers[name]
    return ApsStreamingTransformerEncoder(enc_layer_cls(**arch_kwargs),
                                          num_layers,
                                          norm=final_norm)
