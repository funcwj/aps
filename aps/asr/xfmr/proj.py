#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.layer import Conv1d, Conv2d, Normalize1d
from aps.libs import Register
from typing import Optional, NoReturn, Dict

XfmrProjLayer = Register("xfmr_proj_layer")


def get_xfmr_proj(proj_name: str,
                  in_features: int,
                  att_dim: int,
                  kwargs: Optional[Dict] = None) -> nn.Module:
    """
    Return projection layers
    """
    if proj_name not in XfmrProjLayer:
        raise ValueError(f"Unsupported projection layer: {proj_name}")
    if kwargs is None:
        return XfmrProjLayer[proj_name](in_features, att_dim)
    else:
        return XfmrProjLayer[proj_name](in_features, att_dim, **kwargs)


@XfmrProjLayer.register("linear")
class LinearProj(nn.Module):
    """
    Linear projection layer before transformer encoders
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int = 512,
                 dropout: float = 0,
                 norm: str = "LN") -> None:
        super(LinearProj, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = Normalize1d(norm, embed_dim)
        self.drop = nn.Dropout(p=dropout)

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
        out = tf.relu(self.drop(inp))
        return out


@XfmrProjLayer.register("conv1d")
class Conv1dProj(nn.Module):
    """
    1d-conv projection layer before transformer encoders
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int = 512,
                 norm: str = "BN",
                 dropout: float = 0,
                 dim: int = 256) -> None:
        super(Conv1dProj, self).__init__()
        self.conv1 = Conv1d(input_size, dim, dropout=dropout, norm=norm)
        self.conv2 = Conv1d(dim, embed_dim, dropout=dropout, norm=norm)

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
            out_len = self.conv2.compute_outp_dim(out_len)
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


@XfmrProjLayer.register("conv2d")
class Conv2dProj(nn.Module):
    """
    2d-conv projection layer before transformer encoders
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int = 512,
                 in_channels: int = 1,
                 conv_channels: int = 256) -> None:
        super(Conv2dProj, self).__init__()
        # kernel size = 3, stride = 2
        self.conv1 = Conv2d(in_channels, conv_channels)
        input_size = self.conv1.compute_outp_dim(input_size, 1)
        self.conv2 = Conv2d(conv_channels, conv_channels)
        input_size = self.conv2.compute_outp_dim(input_size, 1)
        self.proj = nn.Linear(input_size * conv_channels, embed_dim)

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
            out_len = self.conv1.compute_outp_dim(inp_len, 0)
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
