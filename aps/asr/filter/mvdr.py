#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

import torch.nn.functional as tf

from aps.asr.base.attention import padding_mask
from aps.asr.base.encoder import PyTorchRNNEncoder
from aps.asr.filter.conv import EnhFrontEnds
from aps.const import EPSILON
from aps.cplx import ComplexTensor
from typing import Optional


def trace(cplx_mat: ComplexTensor) -> ComplexTensor:
    """
    Return trace of a complex matrices
    """
    mat_size = cplx_mat.size()
    diag_index = th.eye(mat_size[-1], dtype=th.bool).expand(*mat_size)
    return cplx_mat.masked_select(diag_index).view(*mat_size[:-1]).sum(-1)


def beamform(weight: ComplexTensor,
             spectrogram: ComplexTensor) -> ComplexTensor:
    """
    Do beamforming
    Args:
        weight: complex, N x C x F
        spectrogram: complex, N x C x F x T (output by STFT)
    Return:
        beam: complex, N x F x T
    """
    return (weight[..., None].conj() * spectrogram).sum(dim=1)


def estimate_covar(mask: th.Tensor,
                   spectrogram: ComplexTensor) -> ComplexTensor:
    """
    Covariance matrices (PSD) estimation
    Args:
        mask: TF-masks (real), N x F x T
        spectrogram: complex, N x C x F x T
    Return:
        covar: complex, N x F x C x C
    """
    # N x F x C x T
    spec = spectrogram.transpose(1, 2)
    # N x F x 1 x T
    mask = mask.unsqueeze(-2)
    # N x F x C x C: einsum("...it,...jt->...ij", spec * mask, spec.conj())
    nominator = (spec * mask) @ spec.conj_transpose(-1, -2)
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

    def _derive_weight(self,
                       Rs: ComplexTensor,
                       Rn: ComplexTensor,
                       u: th.Tensor,
                       eps: float = 1e-5) -> ComplexTensor:
        """
        Compute mvdr beam weights
        Args:
            Rs, Rn: speech & noise covariance matrices, N x F x C x C
            u: reference selection vector, N x C
        Return:
            weight: N x F x C
        """
        C = Rn.shape[-1]
        I = th.eye(C, device=Rn.device, dtype=Rn.dtype)
        Rn = Rn + I * eps
        # N x F x C x C
        Rn_inv = Rn.inverse()
        # N x F x C x C: einsum("...ij,...jk->...ik", Rn_inv, Rs)
        Rn_inv_Rs = Rn_inv @ Rs
        # N x F
        tr_Rn_inv_Rs = trace(Rn_inv_Rs) + eps
        # N x F x C: einsum("...fnc,...c->...fn", Rn_inv_Rs, u)
        Rn_inv_Rs_u = (Rn_inv_Rs * u).sum(-1)
        # N x F x C
        weight = Rn_inv_Rs_u / tr_Rn_inv_Rs[..., None]
        return weight

    def _process_mask(self, mask: th.Tensor, x_len: th.Tensor) -> th.Tensor:
        """
        Process mask estimated by networks
        """
        if mask is None:
            return mask
        if x_len is not None:
            zero_mask = padding_mask(x_len)  # N x T
            mask = th.masked_fill(mask, zero_mask[..., None], 0)
        if self.mask_norm:
            max_abs = th.norm(mask, float("inf"), dim=1, keepdim=True)
            mask = mask / (max_abs + EPSILON)
        mask = th.transpose(mask, 1, 2)
        return mask

    def forward(self,
                mask_s: th.Tensor,
                x: ComplexTensor,
                mask_n: Optional[th.Tensor] = None,
                x_len: Optional[th.Tensor] = None) -> ComplexTensor:
        """
        Args:
            mask_s: real TF-masks (speech), N x T x F
            x: noisy complex spectrogram, N x C x F x T
            mask_n: real TF-masks (noise), N x T x F
        Return:
            y: enhanced complex spectrogram N x T x F
        """
        # N x F x T
        mask_s = self._process_mask(mask_s, x_len=x_len)
        mask_n = self._process_mask(mask_n, x_len=x_len)
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

    def __init__(self, num_bins: int, att_dim: int) -> None:
        super(ChannelAttention, self).__init__()
        self.proj = nn.Linear(num_bins, att_dim)
        self.gvec = nn.Linear(att_dim, 1)

    def forward(self, Rs: ComplexTensor) -> th.Tensor:
        """
        Args:
            Rs: complex, N x F x C x C
        Return:
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
        return tf.softmax(gvec.squeeze(-1), -1)


@EnhFrontEnds.register("rnn_mask_mvdr")
class RNNMaskMvdr(nn.Module):
    """
    Mask based MVDR method. The masks are estimated using simple RNN networks
    """

    def __init__(self,
                 enh_input_size: int,
                 num_bins: int = 257,
                 rnn_inp_proj: int = None,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 hidden_size: int = 640,
                 bidirectional: bool = True,
                 mask_net_noise: bool = True,
                 mvdr_att_dim: int = 512,
                 mask_norm: bool = True):
        super(RNNMaskMvdr, self).__init__()
        # TF-mask estimation network
        self.mask_net = PyTorchRNNEncoder(enh_input_size,
                                          num_bins *
                                          2 if mask_net_noise else num_bins,
                                          input_project=rnn_inp_proj,
                                          rnn=rnn,
                                          num_layers=num_layers,
                                          hidden=hidden_size,
                                          dropout=dropout,
                                          bidirectional=bidirectional,
                                          non_linear="sigmoid")
        # MVDR beamformer
        self.mvdr_net = MvdrBeamformer(num_bins,
                                       att_dim=mvdr_att_dim,
                                       mask_norm=mask_norm)
        self.mask_net_noise = mask_net_noise

    def forward(self,
                feats: th.Tensor,
                cstft: ComplexTensor,
                eps: float = 1e-5,
                inp_len: Optional[th.Tensor] = None) -> ComplexTensor:
        """
        Args:
            inp (Tensor, ComplexTensor):
                1) features for mask estimation, N x T x F
                2) Complex STFT for doing mvdr, N x C x F x T
        Return:
            enh (ComplexTensor): N x T x F
        """
        # TF-mask estimation: N x T x F
        mask, _ = self.mask_net(feats, inp_len)
        if self.mask_net_noise:
            mask_s, mask_n = th.chunk(mask, 2, dim=-1)
        else:
            mask_s, mask_n = mask, None
        # mvdr beamforming: N x T x F
        enh = self.mvdr_net(mask_s, cstft, x_len=inp_len, mask_n=mask_n)
        return enh
