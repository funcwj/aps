#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F
import torch_complex.functional as cF

EPSILON = th.finfo(th.float32).min


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
    def __init__(self, num_bins, att_dim):
        super(MvdrBeamformer, self).__init__()
        self.ref = ChannelAttention(num_bins, att_dim)

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
        Rn_inv_Rs = cF.einsum("...ij,...jk->...ij", [Rn.inverse(), Rs])
        # N x F
        tr_Rn_inv_Rs = trace(Rn_inv_Rs)
        # N x F x C
        Rn_inv_Rs_u = cF.einsum("...fnc,...c->...fn", [Rn_inv_Rs, u])
        # N x F x C
        weight = Rn_inv_Rs_u / (tr_Rn_inv_Rs[..., None] + eps)
        return weight

    def forward(self, m, x, norm=True):
        """
        args:
            m: real TF-masks, N x T x F
            x: noisy complex spectrogram, N x C x F x T
        return:
            y: enhanced complex spectrogram N x T x F
        """
        if norm:
            max_abs, _ = th.max(m, dim=0, keepdim=True)
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
