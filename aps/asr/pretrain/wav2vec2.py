#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Tuple, Optional
from aps.transform.asr import TFTransposeTransform
from aps.asr.xfmr.encoder import TransformerEncoder


class FeatureEncoderLayer(nn.Module):
    """
    Convolution feature encoder layer in wav2vec2
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 kernel: int,
                 bias: bool = False,
                 dropout: float = 0,
                 use_layernorm: bool = True):
        super(FeatureEncoderLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel,
                              stride,
                              bias=bias)
        self.drop = nn.Dropout(p=dropout)
        self.ln = nn.Sequential(
            TFTransposeTransform(), nn.LayerNorm(out_channels),
            TFTransposeTransform()) if use_layernorm else None

    def forward(self, inp: th.Tensor) -> th.Tensor:
        # N x C x T
        out = self.drop(self.conv(inp))
        if self.ln:
            out = self.ln(out)
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
                 layernorm: bool = True):
        super(FeatureEncoder, self).__init__()
        assert len(kernel) == num_layers
        assert len(stride) == num_layers
        encoder = [
            FeatureEncoderLayer(1 if i == 0 else channels,
                                channels,
                                stride[i],
                                kernel[i],
                                dropout=dropout,
                                use_layernorm=layernorm and i != num_layers - 1)
            for i in range(num_layers)
        ]
        self.encoder = nn.Sequential(*encoder)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        # N x 1 x S => N x C x T
        return self.encoder(inp[:, None, :])


class Quantizer(nn.Module):

    def __init__(self):
        pass
