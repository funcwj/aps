#!/usr/bin/env python

# wujian@2020

import numpy as np
import torch as th
import torch.nn as nn

import torch_complex.functional as cf

from .task import Task

EPSILON = np.finfo(np.float32).eps

__all__ = ["UnsuperEnhTask"]


class UnsuperEnhTask(Task):
    """
    Unsupervised enhancement using ML functions
    """
    def __init__(self, nnet, eps=EPSILON):
        super(UnsuperEnhTask, self).__init__(nnet)
        self.eps = eps

    def estimate_covar(self, mask, obs):
        """
        Covariance matrices estimation
        Args:
            mask (Tensor): N x F x T
            obs (ComplexTensor): N x F x C x T
        Return:
            covar (ComplexTensor): N x F x C x C
        """
        _, _, C, _ = obs.shape
        # N x F x 1 x T
        mask = mask.unsqueeze(-2)
        # N x F x C x C
        nominator = cf.einsum("...it,...jt->...ij", [obs * mask, obs.conj()])
        # N x F x 1 x 1
        denominator = th.clamp(mask.sum(-1, keepdims=True), min=self.eps)
        # N x F x C x C
        Bk = C * nominator / denominator
        # N x F x C x C
        Bk = (Bk + Bk.transpose(-1, -2).conj()) / 2
        return Bk

    def det(self, Bk):
        """
        Compute determinant of the hermitian matrices
        Args:
            Bk (Tensor): N x F x C x C
        Return:
            det (Tensor): N x F
        """
        # N x F x C x 2C
        m = th.cat([Bk.real, -Bk.imag], -1)
        # N x F x C x 2C
        n = th.cat([Bk.imag, Bk.real], -1)
        # N x F x 2C x 2C
        Rk = th.cat([m, n], -2)
        # N x F x 2C
        # eigenvectors=False can not backward error
        ev, _ = th.symeig(Rk, eigenvectors=True)
        # N x F x C
        det = th.cumprod(ev[..., ::2], dim=-1)
        # N x F, non-negative
        det = th.clamp(det[..., -1], min=self.eps)
        return det

    def log_pdf(self, mask, obs):
        """
        Compute log-pdf of the cacgmm distributions
        Args:
            mask (Tensor): N x F x T
            obs (ComplexTensor): N x F x C x T
        Return:
            log_pdf (Tensor)
        """
        _, _, C, _ = obs.shape
        # N x F x C x C
        Bk = self.estimate_covar(mask, obs)
        # add to diag
        I = th.eye(C, device=Bk.device, dtype=Bk.dtype)
        Bk = Bk + I * self.eps
        # N x F
        Dk = self.det(Bk)
        # N x F x C x C
        Bk_inv = Bk.inverse()
        # N x F x T
        K = cf.einsum("...xt,...xy,...yt->...t", [obs.conj(), Bk_inv, obs])
        K = th.clamp(K.real, min=self.eps)
        # N x F x T
        log_pdf = -C * th.log(K) - th.log(Dk[..., None])
        # N x F x T
        return log_pdf

    def forward(self, egs, **kwargs):
        # mag, pha: N x C x F x T
        # ms: N x T x F
        obs, ms = self.nnet(egs["mix"])
        # N x F x C x T
        obs = obs.transpose(1, 2)
        # N x F x T
        ms = ms.transpose(-1, -2)
        # N x F x T
        ps = self.log_pdf(ms, obs)
        pn = self.log_pdf(1 - ms, obs)
        # N x F x T
        log_pdf = th.log((th.exp(ps) + th.exp(pn)) * 0.5)
        # to maxinmum log_pdf
        loss = -th.mean(log_pdf)
        return loss, None