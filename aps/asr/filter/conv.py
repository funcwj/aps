# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Convolution based multi-channel front-end processing
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Union

from aps.transform.utils import mel_filter
from aps.asr.base.encoder import PyTorchRNNEncoder
from aps.libs import Register
from aps.cplx import ComplexTensor

EnhFrontEnds = Register("enh_filter")


class ComplexConvXd(nn.Module):
    """
    Complex convolution layer
    """

    def __init__(self, conv_ins, *args, **kwargs):
        super(ComplexConvXd, self).__init__()
        self.real = conv_ins(*args, **kwargs)
        self.imag = conv_ins(*args, **kwargs)

    def forward(self,
                x: ComplexTensor,
                add_abs: bool = False,
                eps: float = 1e-5) -> Union[ComplexTensor, th.Tensor]:
        # x: complex tensor
        assert isinstance(x, ComplexTensor)
        xr, xi = x.real, x.imag
        br = self.real(xr) - self.imag(xi)
        bi = self.real(xi) + self.imag(xr)
        if not add_abs:
            return ComplexTensor(br, bi)
        else:
            return (br**2 + bi**2 + eps)**0.5


class ComplexConv1d(ComplexConvXd):
    """
    Complex 1D convolution layer
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConv1d, self).__init__(nn.Conv1d, *args, **kwargs)


class ComplexConv2d(ComplexConvXd):
    """
    Complex 2D convolution layer
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__(nn.Conv2d, *args, **kwargs)


@EnhFrontEnds.register("time_invar")
class TimeInvariantFilter(nn.Module):
    """
    Time invariant convolutional front-end (eq beamformer)
    """

    def __init__(self,
                 num_bins: int = 257,
                 weight: Optional[str] = None,
                 num_channels: int = 4,
                 spatial_filters: int = 8,
                 spectra_filters: int = 80,
                 spectra_init: str = "random",
                 batchnorm: bool = True,
                 apply_log: bool = True) -> None:
        super(TimeInvariantFilter, self).__init__()
        if spectra_init not in ["mel", "random"]:
            raise ValueError(f"Unsupported init method: {spectra_init}")
        # conv.weight: spatial_filters*num_bins x num_channels
        self.conv = ComplexConv1d(num_bins,
                                  spatial_filters * num_bins,
                                  num_channels,
                                  groups=num_bins,
                                  padding=0,
                                  bias=False)
        if weight:
            # weight: 2 x spatial_filters x num_channels x num_bins
            w = th.load(weight)
            if w.shape[1] != spatial_filters:
                raise RuntimeError(f"Number of beam got from {w.shape[1]} " +
                                   f"don't match parameter {spatial_filters}")
            if w.shape[2] != num_channels:
                raise RuntimeError(
                    f"Number of channels got from {w.shape[2]} " +
                    f"don't match parameter {num_channels}")
            # weight: 2 x spatial_filters x num_bins x num_channels
            w = w.transpose(-1, -2)
            # 2 x spatial_filters*num_bins x 1 x num_channels
            w = w.view(2, -1, 1, num_channels)
            # init
            self.conv.real.data = w[0]
            self.conv.imag.data = w[1]
        self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        if spectra_init == "mel":
            mel_weight = mel_filter(None, num_bins=num_bins)
            self.proj.weight.data = mel_weight
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.B = spatial_filters
        self.C = num_channels
        self.apply_log = apply_log

    def forward(self, x: th.Tensor, eps: float = 1e-5) -> th.Tensor:
        """
        Args:
            x: N x C x F x T, complex tensor
        Return:
            y: N x B x T x ..., enhanced features
        """
        N, C, F, T = x.shape
        if C != self.C:
            raise RuntimeError(f"Expect input channel {self.C}, but {C}")
        # N x C x F x T => N x T x F x C
        x = x.transpose(1, 3)
        x = x.contiguous()
        # NT x F x C
        x = x.view(-1, F, C)
        # NT x FB x 1
        b = self.conv(x, add_abs=True, eps=eps)
        # NT x F x B
        b = b.view(-1, F, self.B)
        # NT x B x F
        b = b.transpose(1, 2)
        # NT x B x D
        f = tf.relu(self.proj(b))
        if self.apply_log:
            # NT x B x D
            f = th.log(f + eps)
        # N x T x B x D
        f = f.view(N, T, self.B, -1)
        # N x B x T x D
        f = f.transpose(1, 2)
        if self.norm:
            f = self.norm(f)
        # N x B x T x D => N x T x B x D
        f = f.transpose(1, 2).contiguous()
        # N x T x BD
        f = f.view(N, T, -1)
        return f


@EnhFrontEnds.register("time_invar_att")
class TimeInvariantAttFilter(nn.Module):
    """
    Time invariant convolutional front-end with attention
    """

    def __init__(self,
                 num_bins: int = 257,
                 weight: Optional[str] = None,
                 num_channels: int = 4,
                 spatial_filters: int = 8,
                 spectra_filters: int = 80,
                 spectra_init: str = "random",
                 query_type: str = "rnn",
                 batchnorm: bool = True,
                 apply_log: bool = True) -> None:
        super(TimeInvariantAttFilter, self).__init__()
        if spectra_init not in ["mel", "random"]:
            raise ValueError(f"Unsupported init method: {spectra_init}")
        if query_type not in ["rnn", "conv"]:
            raise ValueError(f"Unsupported query type: {query_type}")

        def beamformer(beam):
            return ComplexConv1d(num_bins,
                                 beam * num_bins,
                                 num_channels,
                                 groups=num_bins,
                                 padding=0,
                                 bias=False)

        if query_type == "rnn":
            self.pred_q = PyTorchRNNEncoder(num_bins,
                                            num_bins,
                                            dropout=0.2,
                                            hidden=512)
        else:
            self.pred_q = beamformer(1)
        self.conv_k = beamformer(spatial_filters)
        self.conv_v = beamformer(spatial_filters)

        if weight:
            # weight: 2 x spatial_filters x num_channels x num_bins
            w = th.load(weight)
            if w.shape[1] != spatial_filters:
                raise RuntimeError(f"Number of beam got from {w.shape[1]} " +
                                   f"don't match parameter {spatial_filters}")
            if w.shape[2] != num_channels:
                raise RuntimeError(
                    f"Number of channels got from {w.shape[2]} " +
                    f"don't match parameter {num_channels}")
            # weight: 2 x spatial_filters x num_bins x num_channels
            w = w.transpose(-1, -2)
            # 2 x spatial_filters*num_bins x 1 x num_channels
            w = w.view(2, -1, 1, num_channels)
            # init
            self.conv_v.real.data = w[0]
            self.conv_v.imag.data = w[1]

        self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        if spectra_init == "mel":
            mel_weight = mel_filter(None, num_bins=num_bins)
            self.proj.weight.data = mel_weight
        self.norm = nn.BatchNorm1d(spectra_filters) if batchnorm else None
        self.B = spatial_filters
        self.C = num_channels
        self.apply_log = apply_log

    def forward(self, x: th.Tensor, eps: float = 1e-5) -> th.Tensor:
        """
        Args:
            x: N x C x F x T, complex tensor
        Return:
            y: N x T x ..., enhanced features
        """
        N, C, F, T = x.shape
        if C != self.C:
            raise RuntimeError(f"Expect input channel {self.C}, but {C}")
        # N x C x F x T => N x T x F x C
        x = x.transpose(1, 3)
        x = x.contiguous()
        # NT x F x C
        x_for_conv = x.view(-1, F, C)
        if isinstance(self.pred_q, ComplexConv1d):
            # NT x F x 1
            bq = self.pred_q(x_for_conv, add_abs=True, eps=eps)
            # N x T x F
            bq = bq.view(N, T, F)
        else:
            # N x T x F
            x_ch0 = (x[..., 0] + eps).abs()
            bq, _ = self.pred_q(x_ch0, None)
            # abs
            bq = tf.relu(bq)
        # NT x FB x 1
        bv = self.conv_v(x_for_conv, add_abs=True, eps=eps)
        bk = self.conv_k(x_for_conv, add_abs=True, eps=eps)
        # N x T x F x B
        bv = bv.view(N, T, F, self.B)
        bk = bk.view(N, T, F, self.B)
        # score: N x T x B
        s = th.sum(bq[..., None] * bk, -2)
        # score: N x 1 x B
        s = th.mean(s, -2, keepdim=True)
        # softmax: N x 1 x B
        w = th.softmax(s, -1)
        # value: N x T x F
        v = th.sum(w[:, None] * bv, -1)
        # proj
        f = tf.relu(self.proj(v))
        # log
        if self.apply_log:
            f = th.log(f + eps)
        # norm
        if self.norm:
            f = f.transpose(1, 2)
            f = self.norm(f)
            f = f.transpose(1, 2)
        # N x T x F
        f = f.contiguous()
        return f


@EnhFrontEnds.register("time_variant")
class TimeVariantFilter(nn.Module):
    """
    Time variant convolutional front-end
    """

    def __init__(self,
                 num_bins: int = 257,
                 num_channels: int = 4,
                 time_reception: int = 11,
                 spatial_filters: int = 8,
                 spectra_filters: int = 80,
                 batchnorm: bool = True) -> None:
        super(TimeVariantFilter, self).__init__()
        self.conv = ComplexConv2d(num_bins,
                                  num_bins * spatial_filters,
                                  (time_reception, num_channels),
                                  groups=num_bins,
                                  padding=((time_reception - 1) // 2, 0),
                                  bias=False)
        self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.B = spatial_filters
        self.C = num_channels

    def forward(self, x: th.Tensor, eps: float = 1e-5) -> th.Tensor:
        """
        Args:
            x: N x C x F x T, complex tensor
        Return:
            y: N x T x ..., enhanced features
        """
        N, C, F, T = x.shape
        if C != self.C:
            raise RuntimeError(f"Expect input channel {self.C}, but {C}")
        # N x C x F x T => N x F x T x C
        x = x.permute(0, 2, 3, 1)
        # N x FB x T x 1
        b = self.conv(x, add_abs=True, eps=eps)
        # N x F x B x T
        b = b.view(N, F, self.B, T)
        # N x T x B x F
        b = b.transpose(1, 3)
        # N x T x B x D
        f = self.proj(b)
        # N x T x B x D
        f = th.log(tf.relu(f) + eps)
        # N x B x T x D
        f = f.transpose(1, 2)
        if self.norm:
            f = self.norm(f)
        # N x B x T x D => N x T x B x D
        f = f.contiguous().transpose(1, 2)
        # N x T x BD
        f = f.view(N, T, -1)
        return f
