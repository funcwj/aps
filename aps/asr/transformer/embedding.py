#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from aps.asr.base.layer import Conv1d, Conv2d
from typing import Union, Tuple, Optional, NoReturn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Reference:
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 rel_enc: bool = False,
                 scale_embed: bool = False) -> None:
        super(PositionalEncoding, self).__init__()
        pos_enc = th.zeros(max_len, embed_dim)
        if rel_enc:
            position = th.arange(max_len - 1, -1, -1, dtype=th.float32)
        else:
            position = th.arange(0, max_len, dtype=th.float32)
        # 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        div_term = th.exp(
            th.arange(0, embed_dim, 2, dtype=th.float32) *
            (-math.log(10000.0) / embed_dim))
        pos_enc[:, 0::2] = th.sin(position[:, None] * div_term)
        pos_enc[:, 1::2] = th.cos(position[:, None] * div_term)
        # Tmax x D
        self.register_buffer("pos_enc", pos_enc)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_scale = embed_dim**0.5 if scale_embed else 1
        self.rel_enc = rel_enc

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Remove "pos_enc" from state dict
        """
        state = super(PositionalEncoding,
                      self).state_dict(destination=destination,
                                       prefix=prefix,
                                       keep_vars=keep_vars)
        if "pos_enc" in state:
            state.pop("pos_enc")
        return state

    def forward(self,
                inp: th.Tensor,
                t: int = 0) -> Union[th.Tensor, Tuple[nn.Parameter, th.Tensor]]:
        """
        Args:
            inp: N x T x D
        Return:
            out: T x N x D (keep same as transformer definition)
        """
        _, T, _ = inp.shape
        # T x D
        abs_enc = self.pos_enc[t:t + T]
        # N x T x D
        inp_scale = inp * self.embed_scale
        # add dropout
        if self.rel_enc:
            out = self.dropout(inp_scale)
            abs_enc = self.dropout(abs_enc)
        else:
            out = self.dropout(inp_scale + abs_enc)
        # T x N x D
        out = out.transpose(0, 1)
        return (abs_enc, out) if self.rel_enc else out


class LinearEmbedding(nn.Module):
    """
    Linear projection embedding
    """

    def __init__(self, input_size: int, embed_dim: int = 512) -> None:
        super(LinearEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def num_frames(self, inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Return the output frame number
        """
        return inp_len

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp: features from asr transform, N x T x F
        Return:
            out: N x T x D
        """
        inp = self.norm(self.proj(inp))
        out = F.relu(inp)
        return out


class Conv1dEmbedding(nn.Module):
    """
    1d-conv embedding
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int = 512,
                 inner_channels: int = 256) -> None:
        super(Conv1dEmbedding, self).__init__()
        self.conv1 = Conv1d(input_size, inner_channels)
        self.conv2 = Conv1d(inner_channels, embed_dim)

    def check_args(self, inp: th.Tensor) -> NoReturn:
        """
        Check shape of the tensor
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv1dEmbedding expect 3/4D tensor, got {inp.dim()} instead")

    def num_frames(self, inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Return the output frame number
        """
        if inp_len is None:
            return None
        else:
            out_len = self.conv1.compute_outp_dim(inp_len)
            out_len = self.conv1.compute_outp_dim(out_len)
            return out_len

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp: features from front-end or asr transform, N x C x T x F or N x T x F
        """
        self.check_args(inp)
        if inp.dim() == 4:
            N, _, T, _ = inp.shape
            # N x T x D x C
            inp = inp.transpose(1, -1)
            inp = inp.contiguous()
            # N x T x DC
            inp = inp.view(N, T, -1)
        out = self.conv2(self.conv1(inp))
        return out


class Conv2dEmbedding(nn.Module):
    """
    2d-conv embedding described in:
    Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int = 512,
                 input_channels: int = 1) -> None:
        super(Conv2dEmbedding, self).__init__()
        inner_channels = embed_dim // 2
        self.conv1 = Conv2d(input_channels, inner_channels)
        input_size = self.conv1.compute_outp_dim(input_size, 1)
        self.conv2 = Conv2d(inner_channels, inner_channels)
        input_size = self.conv1.compute_outp_dim(input_size, 1)
        self.proj = nn.Linear(input_size * inner_channels, embed_dim)

    def check_args(self, inp: th.Tensor) -> NoReturn:
        """
        Check shape of the tensor
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv2dEmbedding expect 3/4D tensor, got {inp.dim()} instead")

    def num_frames(self, inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Return the output frame number
        """
        if inp_len is None:
            return None
        else:
            out_len = self.conv2.compute_outp_dim(inp_len, 0)
            out_len = self.conv2.compute_outp_dim(out_len, 0)
            return out_len

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp: N x T x F (from asr transform) or N x C x T x F (from front-end processing)
        """
        self.check_args(inp)
        # N x 1 x T x F => N x A x T' x F'
        out = self.conv1(inp[:, None] if inp.dim() == 3 else inp)
        # N x A x T' x F'
        out = self.conv2(out)
        # N x T' x A x F'
        out = out.transpose(1, 2)
        N, T, _, _ = out.shape
        out = out.contiguous()
        out = out.view(N, T, -1)
        # N x T x D
        out = self.proj(out)
        return out


class IOEmbedding(nn.Module):
    """
    Kinds of feature embedding layer for ASR tasks
        1) Linear transform
        2) Conv1d transform
        3) Conv2d transform
        4) Sparse transform
    """

    def __init__(self,
                 embed_type: str,
                 feature_dim: int,
                 embed_dim: int = 512,
                 dropout: float = 0.1,
                 other_opts: int = -1,
                 scale_embed: bool = False,
                 pos_enc: bool = True,
                 rel_enc: bool = False) -> None:
        super(IOEmbedding, self).__init__()
        if embed_type == "linear":
            self.embed = LinearEmbedding(feature_dim, embed_dim=embed_dim)
        elif embed_type == "conv2d":
            self.embed = Conv2dEmbedding(
                feature_dim,
                embed_dim=embed_dim,
                input_channels=1 if other_opts <= 0 else other_opts)
        elif embed_type == "conv1d":
            self.embed = Conv1dEmbedding(
                feature_dim,
                embed_dim=embed_dim,
                inner_channels=embed_dim if other_opts <= 0 else other_opts)
        elif embed_type == "sparse":
            self.embed = nn.Embedding(feature_dim, embed_dim)
        elif embed_type == "none":
            self.embed = None
        else:
            raise RuntimeError(f"Unsupported embedding type: {embed_type}")
        if pos_enc:
            self.posencode = PositionalEncoding(embed_dim,
                                                dropout=dropout,
                                                rel_enc=rel_enc,
                                                scale_embed=scale_embed,
                                                max_len=6000)
        else:
            self.posencode = None

    def num_frames(self, inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Return the output frame number
        """
        if not hasattr(self.embed, "num_frames"):
            raise RuntimeError(
                "Can not call num_frames as that function in self.embed")
        return self.embed.num_frames(inp_len)

    def forward(self, inp: th.Tensor, t: int = 0) -> th.Tensor:
        """
        Args:
            inp: N x T x F (from asr transform)
        Return:
            out: T' x N x F (to feed transformer)
        """
        if self.embed:
            out = self.embed(inp)
        else:
            out = inp
        if self.posencode:
            out = self.posencode(out, t=t)
        else:
            out = out.transpose(0, 1)
        return out
