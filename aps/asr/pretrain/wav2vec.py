#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf


class FeatureEncoderLayer(nn.Module):
    """
    Convolution feature encoder layer in wav2vec
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 kernel: int,
                 bias: bool = False,
                 dropout: float = 0,
                 norm: str = "layer"):
        super(FeatureEncoderLayer, self).__init__()
        assert norm in ["layer", "instance", "none"]
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel,
                              stride,
                              bias=bias)
        self.drop = nn.Dropout(p=dropout)
        # LayerNorm
        if norm == "none":
            self.norm = None
        else:
            self.norm = nn.GroupNorm(1 if norm == "layer" else out_channels,
                                     out_channels)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        # N x C x T
        out = self.drop(self.conv(inp))
        if self.norm:
            out = self.norm(out)
        return tf.gelu(out)


class FeatureEncoder(nn.Module):
    """
    Feature encoder for wav2vec2 (stack of FeatureEncoderLayer)
    """

    def __init__(self,
                 num_layers: int = 5,
                 channels: int = 512,
                 kernel: Tuple[int] = [10, 8, 4, 4, 4],
                 stride: Tuple[int] = [5, 4, 2, 2, 2],
                 dropout: float = 0,
                 norm: str = "layer"):
        super(FeatureEncoder, self).__init__()
        assert len(kernel) == num_layers
        assert len(stride) == num_layers
        encoder = [
            FeatureEncoderLayer(1 if i == 0 else channels,
                                channels,
                                stride[i],
                                kernel[i],
                                dropout=dropout,
                                norm=norm) for i in range(num_layers)
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        # N x 1 x S => N x C x T
        return self.encoder(inp[:, None, :])
