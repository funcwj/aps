#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Dict
from aps.libs import Register
from aps.asr.transformer.impl import RelMultiheadAttention
from aps.asr.transformer.impl import ApsTransformerEncoderLayer, ApsTransformerEncoder, ApsConformerEncoderLayer

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
                 lctx: int = 1,
                 chunk: int = 1):
        super(StreamingRelMultiheadAttention, self).__init__(embed_dim,
                                                             num_heads,
                                                             dropout=dropout,
                                                             bias=bias)
        assert chunk + lctx > 1
        self.lctx = lctx * chunk
        self.chunk = chunk
        self.reset()

    def reset(self):
        self.init = True
        self.cache_q = th.tensor(0)
        self.cache_k = th.tensor(0)
        self.cache_v = th.tensor(0)

    def step(self, chunk: th.Tensor, key_rel_pose: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            key_rel_pose (Tensor): ... x D
        Return:
            chunk (Tensor): T x N x E
        """
        num_frames = chunk.shape[0]
        if not self.init:
            chunk = chunk[-self.chunk:]
        # C x N x H x D
        query, key, value = self.inp_proj(chunk, chunk, chunk)
        # T x N x H x D
        if not self.init and self.lctx:
            query = th.cat([self.cache_q, query], 0)
            value = th.cat([self.cache_v, value], 0)
            key = th.cat([self.cache_k, key], 0)
        C = query.shape[0]
        # T x N x H x T
        term_a = th.einsum("lnhd,snhd->lnhs", query, key)
        rel_pose = key_rel_pose[-C:, -C:]
        term_b = th.matmul(query, rel_pose[:, None].transpose(-1, -2))
        # T x N x H x T
        logit = term_a + term_b
        # T x N x E
        context, weight = self.context_weight(logit,
                                              value,
                                              attn_mask=None,
                                              key_padding_mask=None)
        context = self.wrap_out(context, weight)[0]
        self.init = False
        if self.lctx:
            self.cache_q = query[-self.lctx:]
            self.cache_v = value[-self.lctx:]
            self.cache_k = key[-self.lctx:]
        return context[-num_frames:]


@StreamingTransformerEncoderLayers.register("xfmr_rel")
class StreamingTransformerRelEncoderLayer(ApsTransformerEncoderLayer):
    """
    Streaming version of Transformer encoder layer using relative position encodings
    """

    def __init__(self,
                 att_dim: int,
                 nhead: int,
                 chunk: int = 1,
                 lctx: int = 1,
                 feedforward_dim: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 activation: str = "relu",
                 pre_norm: bool = False) -> None:
        self_attn = StreamingRelMultiheadAttention(att_dim,
                                                   nhead,
                                                   lctx=lctx,
                                                   chunk=chunk,
                                                   dropout=att_dropout)
        super(StreamingTransformerRelEncoderLayer,
              self).__init__(att_dim,
                             self_attn,
                             feedforward_dim=feedforward_dim,
                             dropout=ffn_dropout,
                             activation=activation,
                             pre_norm=pre_norm)
        self.reset()

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


@StreamingTransformerEncoderLayers.register("cfmr_rel")
class StreamingConformerRelEncoderLayer(ApsConformerEncoderLayer):
    """
    Streaming version of Conformer encoder layer using relative position encodings
    """

    def __init__(self,
                 att_dim: int,
                 nhead: int,
                 lctx: int = 1,
                 chunk: int = 1,
                 feedforward_dim: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 kernel_size: int = 15,
                 pre_norm: bool = False,
                 macaron: bool = True,
                 activation: str = "swish") -> None:
        self_attn = StreamingRelMultiheadAttention(att_dim,
                                                   nhead,
                                                   lctx=lctx,
                                                   chunk=chunk,
                                                   dropout=att_dropout)
        super(StreamingConformerRelEncoderLayer,
              self).__init__(att_dim,
                             self_attn,
                             feedforward_dim=feedforward_dim,
                             dropout=ffn_dropout,
                             activation=activation,
                             kernel_size=kernel_size,
                             macaron=macaron,
                             pre_norm=pre_norm,
                             casual_conv1d=True)
        self.reset()

    def reset(self):
        self.self_attn.reset()
        self.cache_conv = th.tensor(0)
        self.cache_frames = 0

    def conv_step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): T x N x D
        Return
            out (Tensor): T x N x D
        """
        num_frames = chunk.shape[0]
        # T x N x F => N x F x T
        chunk = th.einsum("tnf->nft", chunk)
        # padding cache
        if self.cache_frames > 0:
            chunk = th.cat([self.cache_conv, chunk], -1)
        # padding zeros
        if self.padding - self.cache_frames > 0:
            chunk = tf.pad(chunk, (self.padding - self.cache_frames, 0))
        self.cache_frames = min(num_frames + self.cache_frames, self.padding)
        self.cache_conv = chunk[..., -self.cache_frames:]
        chunk = self.convolution(chunk)
        # N x F x T => T x N x F
        chunk = th.einsum("nft->tnf", chunk)
        return chunk[-num_frames:]

    def step(self, chunk: th.Tensor, inj_pose: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            inj_pose (Tensor): ... x D
        Return:
            chunk (Tensor): T x N x E
        """
        # 1) FFN
        if self.feedforward1 is not None:
            if self.pre_norm:
                chunk = self.feedforward1(
                    self.norm_ffn1(chunk)) * self.macaron_factor + chunk
            else:
                chunk = self.norm_ffn1(
                    self.feedforward1(chunk) * self.macaron_factor + chunk)
        # 2) MHSA
        inp = self.norm_attn(chunk) if self.pre_norm else chunk
        att = self.self_attn.step(inp, inj_pose)
        chunk = chunk + self.dropout(att)
        # 3) CNN + FFN
        if self.pre_norm:
            src = self.conv_step(self.norm_conv(chunk)) + chunk
            out = self.feedforward2(
                self.norm_ffn2(src)) * self.macaron_factor + src
        else:
            src = self.conv_step(self.norm_attn(chunk)) + chunk
            src = self.norm_conv(src)
            out = self.norm_ffn2(
                self.feedforward2(src) * self.macaron_factor + src)
        return out


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
        self.reset()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def step(self, chunk: th.Tensor, inj_pose: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): T x N x E
            inj_pose (Tensor): T x D
        Return:
            out (Tensor): T x N x E
        """
        out = chunk

        for mod in self.layers:
            out = mod.step(out, inj_pose)

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
