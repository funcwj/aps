#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.component import Normalize1d
from aps.asr.base.encoder import Conv1dEncoder, Conv2dEncoder
from aps.libs import Register
from typing import Optional, Tuple, Union, List

XfmrProjLayer = Register("xfmr_proj_layer")

NoneOrTensor = Optional[th.Tensor]
ProjOutputType = Tuple[th.Tensor, NoneOrTensor]


def get_xfmr_proj(proj_name: str, in_features: int, att_dim: int,
                  **kwargs) -> nn.Module:
    """
    Return projection layers
    """
    if proj_name not in XfmrProjLayer:
        raise ValueError(f"Unsupported projection layer: {proj_name}")
    return XfmrProjLayer[proj_name](in_features, att_dim, **kwargs)


@XfmrProjLayer.register("linear")
class LinearProj(nn.Module):
    """
    Linear projection layer before transformer encoders
    """

    def __init__(self,
                 input_size: int,
                 embed_dim: int,
                 dropout: float = 0.0,
                 norm: str = "LN") -> None:
        super(LinearProj, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = Normalize1d(norm, embed_dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, inp: th.Tensor, inp_len: NoneOrTensor) -> ProjOutputType:
        """
        Args:
            inp: features from asr transform, N x T x F
        Return:
            out: N x T x D
        """
        inp = self.norm(self.proj(inp))
        out = tf.relu(self.drop(inp))
        return out, inp_len


@XfmrProjLayer.register("conv1d")
class Conv1dProj(nn.Module):
    """
    1d-conv projection layer before transformer encoders
    """
    Conv1dParam = Union[List[int], int]

    def __init__(self,
                 input_size: int,
                 embed_dim: int,
                 norm: str = "BN",
                 dropout: float = 0.0,
                 dim: int = 256,
                 kernel: Conv1dParam = 3,
                 stride: Conv1dParam = 2,
                 num_layers: int = 2,
                 for_streaming: bool = False) -> None:
        super(Conv1dProj, self).__init__()
        # generally num_layers should not be too large
        assert num_layers in [2, 3, 4]
        self.conv = Conv1dEncoder(input_size,
                                  embed_dim,
                                  dim=dim,
                                  norm=norm,
                                  num_layers=num_layers,
                                  dropout=dropout,
                                  kernel=kernel,
                                  stride=stride,
                                  for_streaming=for_streaming)

    def forward(self, inp: th.Tensor, inp_len: NoneOrTensor) -> ProjOutputType:
        """
        Args:
            inp: features from front-end or asr transform, N x C x T x F or N x T x F
        """
        if inp.dim() == 4:
            N, _, T, _ = inp.shape
            # N x T x D x C
            inp = inp.transpose(1, -1)
            inp = inp.contiguous()
            # N x T x DC
            inp = inp.view(N, T, -1)
        return self.conv(inp, inp_len)


@XfmrProjLayer.register("conv2d")
class Conv2dProj(nn.Module):
    """
    2d-conv projection layer before transformer encoders
    """
    Conv2dParam = Union[List[int], int, List[Tuple[int]]]

    def __init__(self,
                 input_size: int,
                 embed_dim: int,
                 norm: str = "BN",
                 kernel: Conv2dParam = 3,
                 stride: Conv2dParam = 2,
                 num_layers: int = 2,
                 in_channels: int = 1,
                 conv_channels: int = 256,
                 for_streaming: bool = False) -> None:
        super(Conv2dProj, self).__init__()
        # generally num_layers should not be too large
        assert num_layers in [2, 3, 4]
        # kernel size = 3, stride = 2
        self.conv = Conv2dEncoder(input_size,
                                  embed_dim,
                                  channel=conv_channels,
                                  in_channels=in_channels,
                                  num_layers=num_layers,
                                  norm=norm,
                                  kernel=kernel,
                                  stride=stride,
                                  for_streaming=for_streaming)

    def forward(self, inp: th.Tensor, inp_len: NoneOrTensor) -> ProjOutputType:
        """
        Args:
            inp: N x T x F (from asr transform) or N x C x T x F (from front-end processing)
        """
        return self.conv(inp[:, None] if inp.dim() == 3 else inp, inp_len)
