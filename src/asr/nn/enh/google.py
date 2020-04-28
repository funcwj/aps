#!/usr/bin/env python

# wujian@2019

import math
import torch as th
import torch.nn as nn

import librosa.filters as filters

import torch.nn.functional as F

from torch_complex.tensor import ComplexTensor


def init_melfilter(num_bins, sr=16000, num_mels=80, fmin=0.0, fmax=None):
    """
    Return mel-filters
    """
    # fmin & fmax
    fmax = sr // 2 if fmax is None else min(fmax, sr // 2)
    # mel-matrix
    mel = filters.mel(sr, (num_bins - 1) * 2,
                      n_mels=num_mels,
                      fmax=fmax,
                      fmin=fmin,
                      htk=True)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


class FixedBeamformer(nn.Module):
    """
    Fixed beamformer as a layer
    """
    def __init__(self,
                 num_beams,
                 num_channels,
                 num_bins,
                 weight=None,
                 requires_grad=False):
        super(FixedBeamformer, self).__init__()
        if weight:
            # (2, num_directions, num_channels, num_bins)
            w = th.load(weight)
            if w.shape[1] != num_beams:
                raise RuntimeError(f"Number of beam got from {w.shape[1]} " +
                                   f"don't match parameter {num_beams}")
            self.init_weight = weight
        else:
            self.init_weight = None
            w = th.zeros(2, num_beams, num_channels, num_bins)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        # (num_directions, num_channels, num_bins, 1)
        self.real = nn.Parameter(w[0].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.imag = nn.Parameter(w[1].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def extra_repr(self):
        B, M, F, _ = self.real.shape
        return (f"num_beams={B}, num_channels={M}, " +
                f"num_bins={F}, init_weight={self.init_weight}, " +
                f"requires_grad={self.requires_grad}")

    def forward(self, x, beam=None, squeeze=False, trans=False, cplx=True):
        """
        args
            x: N x C x F x T, complex tensor
            beam: N
        return
            br, bi: N x B x F x T or N x F x T
        """
        r, i = x.real, x.imag
        if r.dim() != i.dim() and r.dim() != 4:
            raise RuntimeError(
                f"FixBeamformer accept 4D tensor, got {r.dim()}")
        if self.real.shape[1] != r.shape[1]:
            raise RuntimeError(f"Number of channels mismatch: "
                               f"{r.shape[1]} vs {self.real.shape[1]}")
        if beam is None:
            # output all the beam
            br = th.sum(r.unsqueeze(1) * self.real, 2) + th.sum(
                i.unsqueeze(1) * self.imag, 2)
            bi = th.sum(i.unsqueeze(1) * self.real, 2) - th.sum(
                r.unsqueeze(1) * self.imag, 2)
        else:
            # output selected beam
            br = th.sum(r * self.real[beam], 1) + th.sum(
                i * self.imag[beam], 1)
            bi = th.sum(i * self.real[beam], 1) - th.sum(
                r * self.imag[beam], 1)
        if squeeze:
            br = br.squeeze()
            bi = bi.squeeze()
        if trans:
            br = br.transpose(-1, -2)
            bi = bi.transpose(-1, -2)
        if cplx:
            return ComplexTensor(br, bi)
        else:
            return br, bi


class _FsBeamformer(nn.Module):
    """
    FS (filter and sum) beamformer
    """
    def __init__(self, frame_len, frame_hop):
        super(_FsBeamformer, self).__init__()
        self.unfold = nn.Unfold((frame_len, 1), stride=frame_hop)
        self.frame_len, self.frame_hop = frame_len, frame_hop

    def num_frames(self, s):
        """
        Work out number of frames
        """
        return (s - self.frame_len) // self.frame_hop + 1


class UnfactedFsBeamformer(_FsBeamformer):
    """
    Unfacted form of FS (filter and sum) beamformer
    """
    def __init__(self,
                 num_taps=400,
                 win_size=560,
                 num_channels=4,
                 num_filters=256,
                 log_compress=True):
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

    def forward(self, x):
        """
        args:
            x: multi-channel audio utterances, N x C x S
        return:
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
                 num_taps=81,
                 win_size=560,
                 num_channels=4,
                 spatial_filters=10,
                 spectra_filters=128,
                 spectra_kernels=400,
                 log_compress=True):
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

    def forward(self, x):
        """
        args:
            x: multi-channel audio utterances, N x C x S
        return:
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
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(in_features, out_features, bias=bias)
        self.imag = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        args:
            x: complex tensor
        return:
            y: complex tensor
        """
        if not isinstance(x, ComplexTensor):
            raise RuntimeError(
                f"Expect ComplexTensor object, got {type(x)} instead")
        r = self.real(x.real) - self.imag(x.imag)
        i = self.real(x.imag) + self.imag(x.real)
        return ComplexTensor(r, i)


class CLPFsBeamformer(nn.Module):
    """
    Complex Linear Projection (CLP) model on frequency-domain
    """
    def __init__(self,
                 num_bins=257,
                 weight=None,
                 batchnorm=True,
                 num_channels=4,
                 spatial_filters=5,
                 spectra_filters=128,
                 spectra_init="random",
                 spectra_complex=True,
                 spatial_maxpool=False):
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
                mel_filter = init_melfilter(num_bins)
                self.proj.weight.data = mel_filter
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.spectra_complex = spectra_complex
        # self.spatial_maxpool = spatial_maxpool

    def forward(self, x, eps=1e-5):
        """
        args:
            x: complex tensor, N x C x F x T
        return:
            y: enhanced features, N x P x G x T
        """
        if not isinstance(x, ComplexTensor):
            raise RuntimeError(
                f"Expect ComplexTensor object, got {type(x)} instead")
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
        # N x P x T x G
        return z