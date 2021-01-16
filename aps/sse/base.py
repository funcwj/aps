#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Union, List, NoReturn

supported_nonlinear = {
    "relu": th.relu,  # [0, +oo]
    "tanh": th.tanh,  # [-oo, +oo]
    "softplus": tf.softplus,  # [0, +oo]
    "sigmoid": th.sigmoid,  # [0, 1]
    "softmax": th.softmax
}

all_nonlinear = ["relu", "tanh", "softplus", "sigmoid", "softmax"]
all_wo_softmax = ["relu", "tanh", "softplus", "sigmoid"]
positive_nonlinear = ["relu", "softplus", "sigmoid", "softmax"]
positive_nonlinear_wo_softmax = ["relu", "softplus", "sigmoid"]
positive_nonlinear_wo_softplus = ["relu", "softplus", "sigmoid", "softmax"]
common_nonlinear = ["relu", "sigmoid"]
bounded_nonlinear = ["sigmoid", "softmax"]
unbounded_nonlinear = ["relu", "tanh", "softplus"]


class SseBase(nn.Module):
    """
    The base class for speech separation & enhancement models

    Args:
        transform (nn.Module): see aps.transform.enh
        training_mode (str): training mode, frequency domain or time domain
    """

    def __init__(self,
                 transform: Optional[nn.Module],
                 training_mode: str = "freq"):
        super(SseBase, self).__init__()
        assert training_mode in ["freq", "time"]
        self.enh_transform = transform
        self.training_mode = training_mode

    def check_args(self,
                   mix: th.Tensor,
                   training: bool = True,
                   valid_dim: List[int] = [2]) -> NoReturn:
        """
        Check arguments during training or inference
        """
        if mix.dim() not in valid_dim:
            supported_dim = "/".join([str(d) for d in valid_dim])
            raise RuntimeError(
                f"{self.__class__.__name__} expects {supported_dim}D " +
                f"tensor ({'training' if training else 'inference'}), " +
                f"got {mix.dim()} instead")

    def infer(self,
              mix: th.Tensor,
              mode: str = "freq") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Used for separation & enhancement inference
        Args:
            mix (Tensor): S or N x S (multi-channel)
        Return:
            Tensor: S
        """
        raise NotImplementedError


class MaskNonLinear(nn.Module):
    """
    Non-linear function for mask activation
    """

    def __init__(self,
                 non_linear: str,
                 enable: str = "all",
                 scale: float = 1,
                 value_clip: Optional[float] = None) -> None:
        super(MaskNonLinear, self).__init__()
        supported_set = {
            "positive": positive_nonlinear,
            "positive_wo_softmax": positive_nonlinear_wo_softmax,
            "positive_wo_softplus": positive_nonlinear_wo_softplus,
            "all": all_nonlinear,
            "all_wo_softmax": all_wo_softmax,
            "bounded": bounded_nonlinear,
            "unbounded": unbounded_nonlinear,
            "common": common_nonlinear
        }
        if non_linear not in supported_set[enable]:
            raise ValueError(f"Unsupported nonlinear: {non_linear}")
        self.func = supported_nonlinear[non_linear]
        self.value_clip = value_clip
        self.scale = scale
        self.non_linear = non_linear

    def forward(self, inp: th.Tensor, **kwargs) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x F or N x F x T or N x S x F x T
        Return:
            out (Tensor): same shape as inp
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"MaskNonLinear expects 3/4D tensor, got {inp.dim()}")
        if self.non_linear == "softmax" and inp.dim() != 4:
            raise RuntimeError(
                f"MaskNonLinear (softmax) expects 4D tensor, got {inp.dim()}")
        out = self.func(inp, **kwargs)
        if self.non_linear == "softmax":
            return out
        else:
            out = out * self.scale
            if self.value_clip is not None:
                out = th.clamp_max(out, self.value_clip)
            return out
