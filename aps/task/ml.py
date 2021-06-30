# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Dict
from aps.task.base import Task
from aps.const import EPSILON
from aps.libs import ApsRegisters
from aps.cplx import ComplexTensor


def hermitian_det(Bk: ComplexTensor, eps: float = EPSILON) -> th.Tensor:
    """
    Compute determinant of the hermitian matrices
    Args:
        Bk (ComplexTensor): N x F x C x C
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
    det = th.clamp(det[..., -1], min=eps)
    return det


def estimate_covar(mask: th.Tensor,
                   obs: ComplexTensor,
                   eps: float = EPSILON) -> ComplexTensor:
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
    # N x F x C x C: einsum("...it,...jt->...ij", spec * mask, spec.conj())
    nominator = (obs * mask) @ obs.conj_transpose(-1, -2)
    # N x F x 1 x 1
    denominator = th.clamp(mask.sum(-1, keepdims=True), min=eps)
    # N x F x C x C
    Bk = C * nominator / denominator
    # N x F x C x C
    Bk = (Bk + Bk.conj_transpose(-1, -2)) / 2
    return Bk


@ApsRegisters.task.register("sse@enh_ml")
class MlEnhTask(Task):
    """
    Unsupervised multi-channel speech enhancement using ML (maximum likelihood) function
    """

    def __init__(self, nnet: nn.Module, eps: float = EPSILON) -> None:
        super(MlEnhTask,
              self).__init__(nnet,
                             description="unsupervised speech enhancement "
                             "using ML objective function")
        self.eps = eps

    def log_pdf(self, mask: th.Tensor, obs: ComplexTensor) -> th.Tensor:
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
        Bk = estimate_covar(mask, obs, eps=self.eps)
        # add to diag
        I = th.eye(C, device=Bk.device, dtype=Bk.dtype)
        Bk = Bk + I * self.eps
        # N x F
        Dk = hermitian_det(Bk, eps=self.eps)
        # N x F x C x C
        Bk_inv = Bk.inverse()
        # N x F x T: einsum("...xt,...xy,...yt->...t", obs.conj(), Bk_inv, obs)
        K = (obs.conj() * (Bk_inv @ obs)).sum(-2)
        K = th.clamp(K.real, min=self.eps)
        # N x F x T
        log_pdf = -C * th.log(K) - th.log(Dk[..., None])
        # N x F x T
        return log_pdf

    def forward(self, egs: Dict) -> Dict:
        """
        Compute ML loss, egs contains (without reference data)
            mix (Tensor): N x C x S
        """
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
        return {"loss": loss}
