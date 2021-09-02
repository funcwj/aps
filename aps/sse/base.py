#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.cplx import ComplexTensor
from typing import Optional, Union, List, NoReturn

all_ = ["none", "relu", "tanh", "softplus", "sigmoid", "softmax"]
all_wo_softmax = ["none", "relu", "tanh", "softplus", "sigmoid"]
positive = ["relu", "softplus", "sigmoid", "softmax"]
positive_wo_softmax = ["relu", "softplus", "sigmoid"]
positive_wo_softplus = ["relu", "sigmoid", "softmax"]
common = ["relu", "sigmoid"]
bounded = ["sigmoid", "softmax"]
unbounded = ["none", "relu", "tanh", "softplus"]


def tf_masking(mix_stft: th.Tensor,
               src_mask: th.Tensor,
               channel: int = 0) -> th.Tensor:
    """
    Do time-frequency masking
    Args:
        mix_stft: STFT of the mixture signal, N x (C) x F x T x 2
        src_mask: TF masks of the source component, N x F x T (real) or N x F x T x 2 (complex)
        channel: for channel selection in packed_stft
    Return:
        enh_stft: STFT of the enhanced signal, N x F x T x 2
    """
    stft_dim = mix_stft.dim()
    mask_dim = src_mask.dim()
    assert stft_dim in [4, 5]
    assert mask_dim in [3, 4]
    if stft_dim == 5:
        mix_stft = mix_stft[:, channel]
    mix_stft = ComplexTensor(mix_stft[..., 0], mix_stft[..., 1])
    # complex mask
    if mask_dim == 4:
        assert src_mask.shape[-1] == 2
        src_mask = ComplexTensor(src_mask[..., 0], src_mask[..., 1])
    enh_stft = mix_stft * src_mask
    return enh_stft.as_real()


def softmax(tensor: th.Tensor) -> th.Tensor:
    return th.softmax(tensor, 0)


def identity(tensor: th.Tensor) -> th.Tensor:
    return tensor


supported_nonlinear = {
    "none": identity,  # [-oo, +oo]
    "relu": th.relu,  # [0, +oo]
    "tanh": th.tanh,  # [-oo, +oo]
    "softplus": tf.softplus,  # [0, +oo]
    "sigmoid": th.sigmoid,  # [0, 1]
    "softmax": softmax,  # [0, 1]
}


class SSEBase(nn.Module):
    """
    The base class for speech separation & enhancement models

    Args:
        transform (nn.Module): see aps.transform.enh
        training_mode (str): training mode, frequency domain or time domain
    """

    def __init__(self,
                 transform: Optional[nn.Module],
                 training_mode: str = "freq"):
        super(SSEBase, self).__init__()
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
                f"Expects {supported_dim}D tensor " +
                f"({'training' if training else 'inference'}), " +
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
        raise NotImplementedError()


class MaskNonLinear(nn.Module):
    """
    Non-linear function for mask activation
    """

    def __init__(self,
                 non_linear: str,
                 enable: str = "all",
                 scale: float = 1,
                 vmax: Optional[float] = None,
                 vmin: Optional[float] = None) -> None:
        super(MaskNonLinear, self).__init__()
        supported_set = {
            "positive": positive,
            "positive_wo_softmax": positive_wo_softmax,
            "positive_wo_softplus": positive_wo_softplus,
            "all": all_,
            "all_wo_softmax": all_wo_softmax,
            "bounded": bounded,
            "unbounded": unbounded,
            "common": common
        }
        if non_linear not in supported_set[enable]:
            raise ValueError(f"Unsupported nonlinear: {non_linear}")
        self.non_linear = supported_nonlinear[non_linear]
        self.max = vmax
        self.min = vmin
        self.scale = scale

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): (S) x N x ... x ...
        Return:
            out (Tensor): same shape as inp
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"MaskNonLinear expects 3/4D tensor, got {inp.dim()}")
        out = self.non_linear(inp) * self.scale
        if self.max is not None:
            out = th.clamp_max(out, self.max)
        if self.min is not None:
            out = th.clamp_min(out, self.min)
        return out
