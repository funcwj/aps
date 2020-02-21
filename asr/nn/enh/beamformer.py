#!/usr/bin/env python

# wujian@2019

import math
import torch as th
import torch.nn as nn

import torch.nn.functional as F
import torch_complex.functional as cF

from torch_complex.tensor import ComplexTensor

from ..las.attention import padding_mask

EPSILON = th.finfo(th.float32).eps


def trace(cplx_mat):
    """
    Return trace of a complex matrices
    """
    mat_size = cplx_mat.size()
    E = th.eye(mat_size[-1], dtype=th.bool).expand(*mat_size)
    return cplx_mat[E].view(*mat_size[:-1]).sum(-1)


def beamform(weight, spectrogram):
    """
    Do beamforming
    args:
        weight: complex, N x C x F
        spectrogram: complex, N x C x F x T (output by STFT)
    return:
        beam: complex, N x F x T
    """
    return (weight[..., None].conj() * spectrogram).sum(dim=1)


def estimate_covar(mask, spectrogram):
    """
    Covariance matrices (PSD) estimation
    args:
        mask: TF-masks (real), N x F x T
        spectrogram: complex, N x C x F x T
    return:
        covar: complex, N x F x C x C
    """
    # N x F x C x T
    spec = spectrogram.transpose(1, 2)
    # N x F x 1 x T
    mask = mask.unsqueeze(-2)
    # N x F x C x C
    nominator = cF.einsum("...it,...jt->...ij", [spec * mask, spec.conj()])
    # N x F x 1 x 1
    denominator = th.clamp(mask.sum(-1, keepdims=True), min=EPSILON)
    # N x F x C x C
    return nominator / denominator


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
            y: N x P x T, enhanced features
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
            y: N x P x F x T, enhanced features
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
                 num_channels=4,
                 spatial_filters=5,
                 spectra_filters=128,
                 spectra_complex=True,
                 spatial_maxpool=False):
        super(CLPFsBeamformer, self).__init__()
        self.beam = FixedBeamformer(spatial_filters,
                                    num_channels,
                                    num_bins,
                                    weight=weight,
                                    requires_grad=True)
        if spectra_complex:
            self.proj = ComplexLinear(num_bins, spectra_filters, bias=False)
        else:
            self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        self.spectra_complex = spectra_complex
        self.spatial_maxpool = spatial_maxpool

    def forward(self, x, eps=1e-5):
        """
        args:
            x: N x C x F x T, complex tensor
        return:
            y: N x P x G x T, enhanced features
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
            # N x P x T x G
            p = (b + eps).abs()
            # N x T x F
            if self.spatial_maxpool:
                p, _ = th.max(p, 1)
            # N x T x G
            w = th.relu(self.proj(p)) + eps
        z = th.log(w)
        # N x P x G x T or N x G x T
        return z.transpose(-1, -2)


class MvdrBeamformer(nn.Module):
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer
    """
    def __init__(self, num_bins, att_dim, mask_norm=True):
        super(MvdrBeamformer, self).__init__()
        self.ref = ChannelAttention(num_bins, att_dim)
        self.mask_norm = mask_norm

    def _derive_weight(self, Rs, Rn, u, eps=1e-5):
        """
        Compute mvdr beam weights
        args:
            Rs, Rn: speech & noise covariance matrices, N x F x C x C
            u: reference selection vector, N x C
        return:
            weight: N x F x C
        """
        C = Rn.shape[-1]
        I = th.eye(C, device=Rn.device, dtype=Rn.dtype)
        Rn = Rn + I * eps
        # N x F x C x C
        Rn_inv = Rn.inverse()
        # N x F x C x C
        Rn_inv_Rs = cF.einsum("...ij,...jk->...ik", [Rn_inv, Rs])
        # N x F
        tr_Rn_inv_Rs = trace(Rn_inv_Rs) + eps
        # N x F x C
        Rn_inv_Rs_u = cF.einsum("...fnc,...c->...fn", [Rn_inv_Rs, u])
        # N x F x C
        weight = Rn_inv_Rs_u / tr_Rn_inv_Rs[..., None]
        return weight

    def forward(self, m, x, xlen=None):
        """
        args:
            m: real TF-masks, N x T x F
            x: noisy complex spectrogram, N x C x F x T
        return:
            y: enhanced complex spectrogram N x T x F
        """
        if xlen is not None:
            # N x T
            zero_mask = padding_mask(xlen)
            m = th.masked_fill(m, zero_mask[..., None], 0)
        if self.mask_norm:
            # max_abs, _ = th.max(m, dim=0, keepdim=True)
            # m = m / (max_abs + EPSILON)
            max_abs = th.norm(m, float("inf"), dim=1, keepdim=True)
            m = m / (max_abs + EPSILON)
        # N x T x F => N x F x T
        masks_s = th.transpose(m, 1, 2)
        # N x F x C x C
        Rs = estimate_covar(masks_s, x)
        Rn = estimate_covar(1 - masks_s, x)
        # N x C
        u = self.ref(Rs)
        # N x F x C
        weight = self._derive_weight(Rs, Rn, u)
        # N x C x F
        weight = weight.transpose(1, 2)
        # N x F x T
        beam = beamform(weight, x)
        return beam.transpose(1, 2)


class ChannelAttention(nn.Module):
    """
    Compute u for mvdr beamforming
    """
    def __init__(self, num_bins, att_dim):
        super(ChannelAttention, self).__init__()
        self.proj = nn.Linear(num_bins, att_dim)
        self.gvec = nn.Linear(att_dim, 1)

    def forward(self, Rs):
        """
        args:
            Rs: complex, N x F x C x C
        return:
            u: real, N x C
        """
        C = Rs.shape[-1]
        I = th.eye(C, device=Rs.device, dtype=th.bool)
        # diag is zero, N x F x C
        Rs = Rs.masked_fill(I, 0).sum(-1) / (C - 1)
        # N x C x A
        proj = self.proj(Rs.abs().transpose(1, 2))
        # N x C x 1
        gvec = self.gvec(th.tanh(proj))
        # N x C
        return F.softmax(gvec.squeeze(-1), -1)
