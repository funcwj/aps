#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn.functional as tf

from itertools import permutations

from aps.task.base import Task

EPSILON = th.finfo(th.float32).eps

__all__ = ["SisnrTask", "SnrTask", "SaTask", "WaTask"]


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


def snr(x, s, eps=1e-8):
    """
    Computer SNR
    Args:
        x (Tensor): separated signal, N x S 
        s (Tensor): reference signal, N x S
    Return:
        snr (Tensor): N
    """
    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate " +
                           f"si-snr, {x.shape} vs {s.shape}")
    return 20 * th.log10(eps + l2norm(s) / (l2norm(x - s) + eps))


class TimeDomainTask(Task):
    """
    Time domain task (to be implemented)
    """
    def __init__(self, nnet, num_spks=2, permute=True, mode="max"):
        super(TimeDomainTask, self).__init__(nnet)
        self.num_spks = num_spks
        self.permute = permute  # use pit or not
        self.mode = mode

    def _objf(self, out, ref):
        """
        Return tensor (N) for each mini-batch
        """
        raise NotImplementedError

    def _perm_objf(self, permute, out, ref):
        """
        Return tensor (P x N) for each permutation and mini-batch
        """
        return sum([self._objf(out[s], ref[t])
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
            loss = self._objf(out, ref)
        else:
            num_spks = len(out)
            if num_spks != self.num_spks:
                raise RuntimeError(f"Got {num_spks} from nnet, " +
                                   f"but registered {self.num_spks}")
            if self.permute:
                # P x N
                loss_mat = th.stack([
                    self._perm_objf(p, out, ref)
                    for p in permutations(range(num_spks))
                ])
                # NOTE: max or min
                if self.mode == "max":
                    loss, _ = th.max(loss_mat, dim=0)
                else:
                    loss, _ = th.min(loss_mat, dim=0)
            else:
                loss = [self._objf(o, r) for o, r in zip(out, ref)]
                loss = sum(loss) / self.num_spks
        if self.mode == "max":
            return -th.mean(loss), None
        else:
            return th.mean(loss), None


class SisnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    """
    def __init__(self, nnet, num_spks=2, permute=True):
        super(SisnrTask, self).__init__(nnet,
                                        num_spks=num_spks,
                                        permute=permute,
                                        mode="max")

    def _objf(self, out, ref):
        return sisnr(out, ref)


class SnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    """
    def __init__(self, nnet, num_spks=2, permute=True):
        super(SnrTask, self).__init__(nnet,
                                      num_spks=num_spks,
                                      permute=permute,
                                      mode="max")

    def _objf(self, out, ref):
        return snr(out, ref)


class WaTask(TimeDomainTask):
    """
    Time domain waveform approximation loss function
    """
    def __init__(self, nnet, objf="L1", num_spks=2, permute=True):
        super(WaTask, self).__init__(nnet,
                                     num_spks=num_spks,
                                     permute=permute,
                                     mode="min")
        # L2 or L1 loss
        self.objf = objf

    def _objf(self, out, ref):
        if self.objf == "L1":
            loss = tf.l1_loss(out, ref, reduction="none")
        else:
            loss = tf.mse_loss(out, ref, reduction="none")
        return loss.sum(-1)


class FreqSaTask(Task):
    """
    Frequenct SA Task (to be implemented)
    """
    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 permute=True,
                 num_spks=2,
                 masking=True):
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(FreqSaTask, self).__init__(nnet, ctx=sa_ctx)
        self.phase_sensitive = phase_sensitive
        self.truncated = truncated
        self.permute = permute
        self.masking = masking
        self.num_spks = num_spks

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        raise NotImplementedError

    def _ref_mag(self, mix_mag, mix_pha, ref):
        """
        Compute reference magnitude for SA
        """
        ref_mag, ref_pha = self.ctx(ref, output="polar")
        if self.truncated is None:
            return ref_mag
        # truncated
        ref_mag = th.min(ref_mag, self.truncated * mix_mag)
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
            loss_mat = self._objf(out[s] * mix_mag if self.masking else out[s],
                                  ref[t],
                                  reduction="none")
            loss_utt = th.sum(loss_mat.mean(-1), -1)  # x N, per-frame
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
            loss = self._objf(mask * mix_mag if self.masking else mask,
                              ref,
                              reduction="sum")
            # per-frame loss
            loss = loss / (N * ref.shape[-1])
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
                    self._objf(m * mix if self.masking else m,
                               r,
                               reduction="sum")
                    for m, mix, r in zip(mask, mix_mag, ref)
                ]
                loss = sum(loss) / (self.num_spks * N)
        return loss, None


class SaTask(FreqSaTask):
    """
    Frequency domain spectrum approximation (MSA or tPSA) loss function
    """
    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 objf="L2",
                 permute=True,
                 num_spks=2,
                 masking=True):
        super(SaTask, self).__init__(nnet,
                                     phase_sensitive=phase_sensitive,
                                     truncated=truncated,
                                     permute=permute,
                                     masking=masking,
                                     num_spks=num_spks)
        # L2 or L1 loss
        self.objf = objf

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        if self.objf == "L1":
            loss = tf.l1_loss(out, ref, reduction=reduction)
        else:
            loss = tf.mse_loss(out, ref, reduction=reduction)
        return loss, None
