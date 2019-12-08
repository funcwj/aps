#!/usr/bin/env python

# wujian@2019

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
    def __init__(self, weight, requires_grad=False):
        super(FixedBeamformer, self).__init__()
        # (2, 18, 7, 257)
        w = th.load(weight)
        # (18, 7, 257, 1)
        self.real = nn.Parameter(w[0].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.imag = nn.Parameter(w[1].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def extra_repr(self):
        B, M, F, _ = self.real.shape
        return (f"num_beams={B}, num_channels={M}, " +
                f"num_bins={F}, requires_grad={self.requires_grad}")

    def forward(self, r, i, beam=None, squeeze=False, trans=False, cplx=True):
        """
        args
            r, i: N x C x F x T
            beam: N
        return
            br, bi: N x B x F x T or N x F x T
        """
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


class MvdrBeamformer(nn.Module):
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer
    """
    def __init__(self, num_bins, att_dim, mask_norm=True):
        super(MvdrBeamformer, self).__init__()
        self.ref = ChannelAttention(num_bins, att_dim)
        self.mask_norm = mask_norm

    def _derive_weight(self, Rs, Rn, u, eps=EPSILON):
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
