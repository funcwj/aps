#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn


class MaskNonLinear(nn.Module):
    """
    Non-linear function for mask activation
    """

    def __init__(self, non_linear, scale=1, clip=None):
        super(MaskNonLinear, self).__init__()
        supported_nonlinear = {
            "relu": tf.relu,
            "sigmoid": th.sigmoid,
            "softmax": tf.softmax
        }
        if non_linear not in supported_nonlinear:
            raise ValueError(f"Unsupported nonlinear: {non_linear}")
        self.func = supported_nonlinear[non_linear]
        self.clip = clip
        self.scale = scale

    def forward(self, inp):
        out = self.func(inp) * self.scale
        if self.clip is not None:
            out = th.clamp_max(out, self.clip)
        return out