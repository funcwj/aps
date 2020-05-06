#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn.functional as tf
import torch_complex.functional as cf

from itertools import permutations
from .task import Task

EPSILON = th.finfo(th.float32).eps

__all__ = ["SisnrTask", "SaTask", "UnsuperEnhTask"]


def sisnr(x, s, eps=1e-8):
    """
    Computer SiSNR
    Args:
        x (Tensor): separated signal, N x S 
        s (Tensor): reference signal, N x S
    Return:
        sisnr (Tensor): N
    """
    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate " +
                           f"si-snr, {x.shape} vs {s.shape}")
    x_zm = x - th.mean(x, dim=-1, keepdim=True)
    s_zm = s - th.mean(s, dim=-1, keepdim=True)
    t = th.sum(x_zm * s_zm, dim=-1,
               keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def hermitian_det(Bk, eps=EPSILON):
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
    det = th.clamp(det[..., -1], min=eps)
    return det


def estimate_covar(mask, obs, eps=EPSILON):
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
    denominator = th.clamp(mask.sum(-1, keepdims=True), min=eps)
    # N x F x C x C
    Bk = C * nominator / denominator
    # N x F x C x C
    Bk = (Bk + Bk.transpose(-1, -2).conj()) / 2
    return Bk


class UnsuperEnhTask(Task):
    """
    Unsupervised enhancement using ML functions
    """
    def __init__(self, nnet, eps=EPSILON):
        super(UnsuperEnhTask, self).__init__(nnet)
        self.eps = eps

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
        Bk = estimate_covar(mask, obs, eps=self.eps)
        # add to diag
        I = th.eye(C, device=Bk.device, dtype=Bk.dtype)
        Bk = Bk + I * self.eps
        # N x F
        Dk = hermitian_det(Bk, eps=self.eps)
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
        return loss, None


class SisnrTask(Task):
    """
    Time domain sisnr loss function
    """
    def __init__(self, nnet, num_spks=2, permute=True):
        super(SisnrTask, self).__init__(nnet)
        self.num_spks = num_spks
        # use pit or not
        self.permute = permute

    def _perm_sisnr(self, permute, out, ref):
        # for one permute
        return sum([sisnr(out[s], ref[t])
                    for s, t in enumerate(permute)]) / len(permute)

    def forward(self, egs, **kwargs):
        """
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        ref = egs["ref"]
        # do separation or enhancement
        # out: Tensor or [Tensor, ...]
        out = self.nnet(egs["mix"])

        if isinstance(out, th.Tensor):
            snr = sisnr(out, ref)
        else:
            num_spks = len(out)
            if num_spks != self.num_spks:
                raise RuntimeError(f"Got {num_spks} from nnet, " +
                                   f"but registered {self.num_spks}")
            if self.permute:
                # P x N
                sisnr_mat = th.stack([
                    self._perm_sisnr(p, out, ref)
                    for p in permutations(range(num_spks))
                ])
                snr, _ = th.max(sisnr_mat, dim=0)
            else:
                snr = [sisnr(o, r) for o, r in zip(out, ref)]
                snr = sum(snr) / self.num_spks
        return -th.mean(snr), None


class SaTask(Task):
    """
    Frequency domain spectrum approximation (MSA or tPSA) loss function
    """
    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 objf="L2",
                 permute=True,
                 num_spks=2):
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(SaTask, self).__init__(nnet, ctx=sa_ctx)
        self.phase_sensitive = phase_sensitive
        self.truncated = truncated
        self.permute = permute
        self.num_spks = num_spks
        # L2 or L1 loss
        self.objf = tf.mse_loss if objf == "L2" else tf.l1_loss

    def _ref_mag(self, mix_mag, mix_pha, ref):
        """
        Compute reference magnitude for SA
        """
        ref_mag, ref_pha = self.ctx(ref, output="polar")
        if self.truncated is None:
            return ref_mag
        # truncated
        ref_mag = th.clamp_min(ref_mag, self.truncated * mix_mag)
        if not self.phase_sensitive:
            return ref_mag
        # use phase-sensitive
        pha_dif = th.clamp(th.cos(ref_pha - mix_pha), min=0)
        return ref_mag * pha_dif

    def _permu_sa(self, permute, mix_mag, out, ref):
        """
        SA computation in permutation mode
        """
        permu_loss = []
        # for one permutation
        for s, t in enumerate(permute):
            # N x F x T
            loss_mat = self.objf(out[s] * mix_mag, ref[t], reduction="none")
            loss_utt = th.sum(loss_mat, (1, 2))  # x N
            permu_loss.append(loss_utt)
        return sum(permu_loss)

    def forward(self, egs, **kwargs):
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix = egs["mix"]
        N = mix.shape[0]
        # do separation or enhancement
        # out: Tensor or [Tensor, ...]
        mask = self.nnet(mix)

        # if multi-channel, use ch0 as reference
        mix_mag, mix_pha = self.ctx(mix[:, 0] if mix.dim() == 3 else mix,
                                    output="polar")

        if isinstance(mask, th.Tensor):
            # F x T
            ref = self._ref_mag(mix_mag, mix_pha, egs["ref"])
            loss = self.objf(mask * mix_mag, ref, reduction="sum")
            loss = loss / N
        else:
            num_spks = len(mask)
            if num_spks != self.num_spks:
                raise RuntimeError(f"Got {num_spks} from nnet, " +
                                   f"but registered {self.num_spks}")
            # for each reference
            ref = [self._ref_mag(mix_mag, mix_pha, r) for r in egs["ref"]]
            if self.permute:
                # P x N
                permu_loss = th.stack([
                    self._permu_sa(p, mix_mag, mask, ref)
                    for p in permutations(range(num_spks))
                ])
                # N
                min_val, _ = th.min(permu_loss, dim=0)
                loss = th.mean(min_val) / num_spks
            else:
                loss = [
                    self.objf(m * mix, r, reduction="sum")
                    for m, mix, r in zip(mask, mix_mag, ref)
                ]
                loss = sum(loss) / (self.num_spks * N)
        return loss, None
