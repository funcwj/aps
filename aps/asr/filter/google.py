#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from aps.transform.utils import init_melfilter
from aps.transform.enh import FixedBeamformer
from aps.asr.filter.conv import EnhFrontEnds
from aps.cplx import ComplexTensor

from typing import Optional


class _FsBeamformer(nn.Module):
    """
    FS (filter and sum) beamformer
    """

    def __init__(self, frame_len: int, frame_hop: int) -> None:
        super(_FsBeamformer, self).__init__()
        self.unfold = nn.Unfold((frame_len, 1), stride=frame_hop)
        self.frame_len, self.frame_hop = frame_len, frame_hop

    def num_frames(self, s: th.Tensor) -> th.Tensor:
        """
        Work out number of frames
        """
        return (s - self.frame_len) // self.frame_hop + 1


class UnfactedFsBeamformer(_FsBeamformer):
    """
    Unfacted form of FS (filter and sum) beamformer
    """

    def __init__(self,
                 num_taps: int = 400,
                 win_size: int = 560,
                 num_channels: int = 4,
                 num_filters: int = 256,
                 log_compress: bool = True) -> None:
        super(UnfactedFsBeamformer, self).__init__(win_size,
                                                   win_size - num_taps)
        self.num_channels = num_channels
        self.log_compress = log_compress
        # fs beamformer
        self.filter = nn.Conv2d(num_channels,
                                num_filters * num_channels, (num_taps, 1),
                                stride=(1, 1),
                                groups=num_channels,
                                bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: multi-channel audio utterances, N x C x S
        Return:
            y: enhanced features, N x P x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"Expect 2/3D tensor, got {x.dim()} instead")
        if x.dim() == 2:
            x = x[None, ...]
        # N x C x S x 1
        x = x[..., None]
        # chunks: N x C x S x 1 => N x CM x T
        c = self.unfold(x)
        # N x C x M x T
        c = c.view(x.shape[0], self.num_channels, self.frame_len, -1)
        # N x CF x M' x T
        f = self.filter(c)
        # N x F x M' x T
        f = sum(th.chunk(f, self.num_channels, 1))
        # max pool, N x F x 1 x T
        y = F.max_pool2d(f, (self.frame_hop + 1, 1), stride=1)
        # non-linear
        y = th.relu(y.squeeze(-2))
        # log
        if self.log_compress:
            y = th.log(y + 0.01)
        return y


class FactedFsBeamformer(_FsBeamformer):
    """
    Facted form of FS (filter and sum) beamformer
    """

    def __init__(self,
                 num_taps: int = 81,
                 win_size: int = 560,
                 num_channels: int = 4,
                 spatial_filters: int = 10,
                 spectra_filters: int = 128,
                 spectra_kernels: int = 400,
                 log_compress: bool = True) -> None:
        super(FactedFsBeamformer, self).__init__(win_size,
                                                 win_size - spectra_kernels)
        self.num_channels = num_channels
        self.log_compress = log_compress
        # spatial filter
        self.spatial = nn.Conv2d(num_channels,
                                 spatial_filters * num_channels, (num_taps, 1),
                                 stride=(1, 1),
                                 groups=num_channels,
                                 bias=False,
                                 padding=((num_taps - 1) // 2, 0))
        # spectra filter
        self.spectra = nn.Conv2d(1,
                                 spectra_filters, (spectra_kernels, 1),
                                 stride=(1, 1),
                                 bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: multi-channel audio utterances, N x C x S
        Return:
            y: enhanced features, N x P x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(f"Expect 2/3D tensor, got {x.dim()} instead")
        if x.dim() == 2:
            x = x[None, ...]
        # N x C x S x 1
        x = x[..., None]
        # chunks: N x C x S x 1 => N x CM x T
        c = self.unfold(x)
        # N x C x M x T
        c = c.view(x.shape[0], self.num_channels, self.frame_len, -1)
        # spatial filter: N x CP x M x T
        f = self.spatial(c)
        # N x P x M x T
        f = sum(th.chunk(f, self.num_channels, 1))
        N, P, M, T = f.shape
        # NP x 1 x M x T
        f = f.view(N * P, 1, M, T)
        # spectra filter: NP x F x M' x T
        w = self.spectra(f)
        # max pool, NP x F x 1 x T
        y = F.max_pool2d(w, (self.frame_hop + 1, 1), stride=1)
        # non-linear
        y = th.relu(y.squeeze(-2))
        # log
        if self.log_compress:
            y = th.log(y + 0.01)
        y = y.view(N, P, -1, T)
        return y


class ComplexLinear(nn.Module):
    """
    Complex linear layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: ComplexTensor) -> ComplexTensor:
        """
        args:
            x: complex tensor
        return:
            y: complex tensor
        """
        assert isinstance(x, ComplexTensor)
        r = self.real(x.real) - self.imag(x.imag)
        i = self.real(x.imag) + self.imag(x.real)
        return ComplexTensor(r, i)


@EnhFrontEnds.register("google_clp")
class CLPFsBeamformer(nn.Module):
    """
    Complex Linear Projection (CLP) model on frequency-domain
    """

    def __init__(self,
                 num_bins: int = 257,
                 weight: Optional[str] = None,
                 batchnorm: bool = True,
                 num_channels: int = 4,
                 spatial_filters: int = 5,
                 spectra_filters: int = 128,
                 spectra_init: str = "random",
                 spectra_complex: bool = True,
                 spatial_maxpool: bool = False) -> None:
        super(CLPFsBeamformer, self).__init__()
        if spectra_init not in ["mel", "random"]:
            raise ValueError(f"Unsupported init method: {spectra_init}")
        self.beam = FixedBeamformer(spatial_filters,
                                    num_channels,
                                    num_bins,
                                    weight=weight,
                                    requires_grad=True)
        if spectra_complex:
            self.proj = ComplexLinear(num_bins, spectra_filters, bias=False)
        else:
            self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
            if spectra_init == "mel":
                mel_filter = init_melfilter(None, num_bins=num_bins)
                self.proj.weight.data = mel_filter
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.spectra_complex = spectra_complex
        # self.spatial_maxpool = spatial_maxpool

    def forward(self, x: ComplexTensor, eps: float = 1e-5) -> th.Tensor:
        """
        Args:
            x: complex tensor, N x C x F x T
        Return:
            y: enhanced features, N x P x G x T
        """
        assert isinstance(x, ComplexTensor)
        if x.dim() not in [3, 4]:
            raise RuntimeError(f"Expect 3/4D tensor, got {x.dim()} instead")
        if x.dim() == 3:
            x = x[None, ...]
        # N x P x T x F
        b = self.beam(x, trans=True, cplx=True)
        if self.spectra_complex:
            # N x P x T x G
            w = self.proj(b)
            # log + abs: N x P x T x G
            w = (w + eps).abs()
        else:
            # N x P x T x F
            p = (b + eps).abs()
            # N x P x T x G
            w = th.relu(self.proj(p)) + eps
        z = th.log(w)
        # N x P x T x G
        if self.norm:
            z = self.norm(z)
        # N x P x T x G => N x T x P x G
        z = z.transpose(1, 2).contiguous()
        # N x T x BD
        z = z.view(*z.shape[:2], -1)
        return z
