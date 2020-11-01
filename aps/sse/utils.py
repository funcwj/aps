#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional


class MaskNonLinear(nn.Module):
    """
    Non-linear function for mask activation
    """

    def __init__(self,
                 non_linear: str,
                 scale: float = 1,
                 clip: Optional[float] = None) -> None:
        super(MaskNonLinear, self).__init__()
        supported_nonlinear = {
            "relu": tf.relu,
            "tanh": tf.tanh,
            "sigmoid": th.sigmoid,
            "softmax": tf.softmax
        }
        if non_linear not in supported_nonlinear:
            raise ValueError(f"Unsupported nonlinear: {non_linear}")
        self.func = supported_nonlinear[non_linear]
        self.clip = clip
        self.scale = scale

    def forward(self, inp: th.Tensor) -> th.Tensor:
        out = self.func(inp) * self.scale
        if self.clip is not None:
            out = th.clamp_max(out, self.clip)
        return out
