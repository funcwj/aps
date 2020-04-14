#!/usr/bin/env python

# wujian@2019

import math
import torch as th
import torch.nn as nn

import torch.nn.functional as F
import torch_complex.functional as cF

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


class MvdrBeamformer(nn.Module):
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer
    """
    def __init__(self, num_bins, att_dim=512, mask_norm=True, eps=1e-5):
        super(MvdrBeamformer, self).__init__()
        self.ref = ChannelAttention(num_bins, att_dim)
        self.mask_norm = mask_norm
        self.eps = eps

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

    def _process_mask(self, mask, xlen):
        """
        Process mask estimated by networks
        """
        if mask is None:
            return mask
        if xlen is not None:
            zero_mask = padding_mask(xlen)  # N x T
            mask = th.masked_fill(mask, zero_mask[..., None], 0)
        if self.mask_norm:
            max_abs = th.norm(mask, float("inf"), dim=1, keepdim=True)
            mask = mask / (max_abs + EPSILON)
        return mask

    def forward(self, mask_s, x, mask_n=None, xlen=None):
        """
        args:
            mask_s: real TF-masks (speech), N x T x F
            x: noisy complex spectrogram, N x C x F x T
            mask_n: real TF-masks (noise), N x T x F
        return:
            y: enhanced complex spectrogram N x T x F
        """
        # N x T x F
        mask_s = self._process_mask(mask_s, xlen=xlen)
        mask_n = self._process_mask(mask_n, xlen=xlen)
        # N x T x F => N x F x T
        mask_s = th.transpose(mask_s, 1, 2)
        # N x F x C x C
        Rs = estimate_covar(mask_s, x)
        Rn = estimate_covar(1 - mask_s if mask_n is None else mask_n, x)
        # N x C
        u = self.ref(Rs)
        # N x F x C
        weight = self._derive_weight(Rs, Rn, u, eps=self.eps)
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
