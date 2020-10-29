#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from itertools import permutations

from aps.task.base import Task
from aps.task.utils import permu_invarint_objf, multiple_objf
from aps.transform.utils import STFT, init_melfilter
from aps.const import EPSILON

__all__ = [
    "SisnrTask", "SnrTask", "WaTask", "LinearFreqSaTask", "LinearTimeSaTask",
    "MelFreqSaTask", "MelTimeSaTask", "ComplexMappingTask"
]


def sisnr(x, s, eps=1e-8, zero_mean=True, non_nagetive=False):
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
    if zero_mean:
        x = x - th.mean(x, dim=-1, keepdim=True)
        s = s - th.mean(s, dim=-1, keepdim=True)
    t = th.sum(x * s, dim=-1,
               keepdim=True) * s / (l2norm(s, keepdim=True)**2 + eps)

    snr_linear = l2norm(t) / (l2norm(x - t) + eps)
    if non_nagetive:
        return 10 * th.log10(1 + snr_linear**2)
    else:
        return 20 * th.log10(eps + snr_linear)


def snr(x, s, eps=1e-8, non_nagetive=False):
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
    snr_linear = l2norm(s) / (l2norm(x - s) + eps)
    if non_nagetive:
        return 10 * th.log10(1 + snr_linear**2)
    else:
        return 20 * th.log10(eps + snr_linear)


def hybrid_objf(out, ref, objf, weight=None, permute=True, permu_num_spks=2):
    """
    Return hybrid loss (pair-wise, permutated or pair-wise + permutated)
    """
    num_branch = len(out)
    if permute:
        # N
        loss = permu_invarint_objf(out[:permu_num_spks], ref[:permu_num_spks],
                                   objf)
        # add residual loss
        if num_branch > permu_num_spks:
            # warnings.warn(f"#Branch: {num_branch} > #Speaker: {permu_num_spks}")
            num_weight = num_branch - (permu_num_spks - 1)
            if weight is None:
                weight = [1 / num_weight] * num_weight
            other_loss = multiple_objf(out[permu_num_spks:],
                                       ref[permu_num_spks:],
                                       objf,
                                       weight=weight[1:])
            loss = weight[0] * loss + other_loss
    else:
        loss = multiple_objf(out, ref, objf, weight=weight)
    return loss


class SepTask(Task):
    """
    Base class for separation & enhancement task
    """

    def __init__(self, nnet, ctx=None, name="unknown", weight=None):
        super(SepTask, self).__init__(nnet, ctx=ctx, name=name)
        if weight is not None:
            self.weight = list(map(float, weight.split(",")))
        else:
            self.weight = None

    def objf(self, out, ref):
        """
        Return tensor (N) for each mini-batch
        """
        raise NotImplementedError

    def transform(self, tensor):
        """
        Transformation on out & ref before calling objf
        """
        raise NotImplementedError


class TimeDomainTask(SepTask):
    """
    Time domain task (to be implemented)
    """

    def __init__(self, nnet, num_spks=2, permute=True, weight=None):
        super(TimeDomainTask, self).__init__(nnet, weight=weight)
        self.num_spks = num_spks
        self.permute = permute  # use pit or not

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
            loss = self.objf(out, ref)
        else:
            if len(out) != len(ref):
                raise RuntimeError(
                    f"Got {len(ref)} references but with {len(out)} outputs")
            loss = hybrid_objf(out,
                               ref,
                               self.objf,
                               weight=self.weight,
                               permute=self.permute,
                               permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


class SisnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    """

    def __init__(self,
                 nnet,
                 num_spks=2,
                 permute=True,
                 weight=None,
                 zero_mean=True,
                 non_nagetive=False):
        super(SisnrTask, self).__init__(nnet,
                                        num_spks=num_spks,
                                        permute=permute,
                                        weight=weight)
        self.zero_mean = zero_mean
        self.non_nagetive = non_nagetive

    def objf(self, out, ref):
        """
        Return negative SiSNR
        """
        return -sisnr(
            out, ref, zero_mean=self.zero_mean, non_nagetive=self.non_nagetive)


class SnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    """

    def __init__(self,
                 nnet,
                 num_spks=2,
                 permute=True,
                 weight=None,
                 non_nagetive=False):
        super(SnrTask, self).__init__(nnet,
                                      num_spks=num_spks,
                                      permute=permute,
                                      weight=weight)
        self.non_nagetive = non_nagetive

    def objf(self, out, ref):
        """
        Return negative SNR
        """
        return -snr(out, ref, non_nagetive=self.non_nagetive)


class WaTask(TimeDomainTask):
    """
    Time domain waveform approximation loss function
    """

    def __init__(self, nnet, objf="L1", num_spks=2, permute=True, weight=None):
        super(WaTask, self).__init__(nnet,
                                     num_spks=num_spks,
                                     permute=permute,
                                     weight=weight)
        # L2 or L1 loss
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out, ref):
        """
        L1 or L2
        """
        loss = self.objf_ptr(out, ref, reduction="none")
        return loss.sum(-1)


class FreqSaTask(SepTask):
    """
    Frequenct SA Task (to be implemented)
    """

    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 permute=True,
                 num_spks=2,
                 masking=True,
                 weight=None):
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(FreqSaTask, self).__init__(nnet, ctx=sa_ctx, weight=weight)
        if not masking and truncated:
            raise ValueError(
                "Conflict parameters: masksing = True while truncated != None")
        self.phase_sensitive = phase_sensitive
        self.truncated = truncated
        self.permute = permute
        self.masking = masking
        self.num_spks = num_spks

    def _ref_mag(self, mix_mag, mix_pha, ref):
        """
        Compute reference magnitude for SA
        """
        ref_mag, ref_pha = self.ctx(ref, output="polar")
        # use phase-sensitive
        if self.phase_sensitive:
            # non-negative
            pha_dif = th.clamp(th.cos(ref_pha - mix_pha), min=0)
            ref_mag = ref_mag * pha_dif
        # truncated
        if self.truncated is not None:
            ref_mag = th.min(ref_mag, self.truncated * mix_mag)
        return ref_mag

    def forward(self, egs, **kwargs):
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix = egs["mix"]
        # do separation or enhancement
        # out: Tensor or [Tensor, ...]
        mask = self.nnet(mix)

        # if multi-channel, use ch0 as reference
        mix_mag, mix_pha = self.ctx(mix[:, 0] if mix.dim() == 3 else mix,
                                    output="polar")

        if isinstance(mask, th.Tensor):
            # F x T
            ref = self._ref_mag(mix_mag, mix_pha, egs["ref"])
            # post processing
            out = self.transform(mask * mix_mag if self.masking else mask)
            ref = self.transform(ref)
            # loss
            loss = self.objf(out, ref)
        else:
            if len(mask) != len(egs["ref"]):
                raise RuntimeError(
                    f"Got {len(egs['ref'])} references but with {len(mask)} outputs"
                )
            # for each reference
            ref = [self._ref_mag(mix_mag, mix_pha, r) for r in egs["ref"]]
            if self.masking:
                out = [m * mix_mag for m in mask]
            else:
                out = mask
            ref = [self.transform(r) for r in ref]
            out = [self.transform(o) for o in out]
            loss = hybrid_objf(out,
                               ref,
                               self.objf,
                               weight=self.weight,
                               permute=self.permute,
                               permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


class LinearFreqSaTask(FreqSaTask):
    """
    Frequency domain linear spectral approximation (MSA or tPSA) loss function
    """

    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 objf="L2",
                 permute=True,
                 num_spks=2,
                 weight=None,
                 masking=True):
        super(LinearFreqSaTask, self).__init__(nnet,
                                               phase_sensitive=phase_sensitive,
                                               truncated=truncated,
                                               permute=permute,
                                               masking=masking,
                                               weight=weight,
                                               num_spks=num_spks)
        # L2 or L1 loss
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out, ref):
        """
        Return loss for each mini-batch
        """
        # out, ref: N x F x T
        loss = self.objf_ptr(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss

    def transform(self, tensor):
        """
        Just return itself
        """
        return tensor


class MelFreqSaTask(FreqSaTask):
    """
    Spectral approximation on mel-filter domain
    """

    def __init__(self,
                 nnet,
                 phase_sensitive=False,
                 truncated=None,
                 weight=None,
                 permute=True,
                 num_spks=2,
                 num_bins=257,
                 masking=True,
                 power_mag=False,
                 num_mels=80,
                 mel_log=False,
                 mel_scale=1,
                 mel_norm=True,
                 sr=16000,
                 fmax=8000):
        super(MelFreqSaTask, self).__init__(nnet,
                                            phase_sensitive=phase_sensitive,
                                            truncated=truncated,
                                            permute=permute,
                                            masking=masking,
                                            weight=weight,
                                            num_spks=num_spks)
        mel = init_melfilter(None,
                             num_bins=num_bins,
                             sr=sr,
                             num_mels=num_mels,
                             fmax=fmax,
                             norm=mel_norm)
        self.mel = nn.Parameter(mel[..., None] * mel_scale, requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def transform(self, tensor):
        """
        Return mel spectrogram
        """
        if self.power_mag:
            tensor = tensor**2
        # N x F x T => N x M x T
        mel = tf.conv1d(tensor, self.mel.to(tensor.device), bias=None)
        if self.log:
            mel = th.log(1 + mel)
        return mel

    def objf(self, out, ref):
        """
        Computer MSE after mel-transform
        """
        loss = tf.mse_loss(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss


class TimeSaTask(SepTask):
    """
    Time domain spectral approximation Task
    """

    def __init__(self,
                 nnet,
                 frame_len=512,
                 frame_hop=256,
                 center=False,
                 window="sqrthann",
                 round_pow_of_two=True,
                 stft_normalized=False,
                 pre_emphasis=0,
                 permute=True,
                 weight=None,
                 num_spks=2):
        # STFT context
        sa_ctx = STFT(frame_len,
                      frame_hop,
                      window=window,
                      center=center,
                      round_pow_of_two=round_pow_of_two,
                      normalized=stft_normalized)
        super(TimeSaTask, self).__init__(nnet, ctx=sa_ctx, weight=weight)
        self.permute = permute
        self.num_spks = num_spks
        self.pre_emphasis = pre_emphasis

    def _stft_mag(self, wav):
        """
        Compute STFT magnitude for SA loss
        """
        # for ASR (do pre-emphasis)
        if self.pre_emphasis > 0:
            wav[:, 1:] = wav[:, 1:] - self.pre_emphasis * wav[:, :-1]
        mag, _ = self.ctx(wav, output="polar")
        return mag

    def forward(self, egs, **kwargs):
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix = egs["mix"]
        # do separation or enhancement
        # spk: Tensor or [Tensor, ...]
        spk = self.nnet(mix)

        if isinstance(spk, th.Tensor):
            # F x T
            spk_mag = self._stft_mag(spk)
            ref_mag = self._stft_mag(egs["ref"])
            # loss (N)
            loss = self.objf(self.transform(spk_mag), self.transform(ref_mag))
        else:
            if len(spk) != len(egs["ref"]):
                raise RuntimeError(
                    f"Got {len(egs['ref'])} reference but with {len(spk)} outputs"
                )
            spk_mag = [self._stft_mag(s) for s in spk]
            # for each reference
            ref_mag = [self._stft_mag(r) for r in egs["ref"]]
            # post
            out = [self.transform(s) for s in spk_mag]
            ref = [self.transform(r) for r in ref_mag]
            loss = hybrid_objf(out,
                               ref,
                               self.objf,
                               weight=self.weight,
                               permute=self.permute,
                               permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


class LinearTimeSaTask(TimeSaTask):
    """
    Time domain linear spectral approximation loss function
    """

    def __init__(self,
                 nnet,
                 frame_len=512,
                 frame_hop=256,
                 center=False,
                 window="sqrthann",
                 round_pow_of_two=True,
                 stft_normalized=False,
                 permute=True,
                 weight=None,
                 num_spks=2,
                 objf="L2"):
        super(LinearTimeSaTask,
              self).__init__(nnet,
                             frame_len=frame_len,
                             frame_hop=frame_hop,
                             window=window,
                             center=center,
                             round_pow_of_two=round_pow_of_two,
                             stft_normalized=stft_normalized,
                             permute=permute,
                             num_spks=num_spks,
                             weight=weight)
        # L2 or L1 loss
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out, ref):
        """
        Return loss for each mini-batch
        """
        loss = self.objf_ptr(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss

    def transform(self, tensor):
        """
        Just return itself
        """
        return tensor


class MelTimeSaTask(TimeSaTask):
    """
    Time domain mel spectral approximation loss function
    """

    def __init__(self,
                 nnet,
                 frame_len=512,
                 frame_hop=256,
                 window="sqrthann",
                 center=False,
                 round_pow_of_two=True,
                 stft_normalized=False,
                 permute=True,
                 weight=None,
                 num_spks=2,
                 num_bins=257,
                 num_mels=80,
                 power_mag=False,
                 mel_log=False,
                 mel_scale=1,
                 mel_norm=True,
                 sr=16000,
                 fmax=7690):
        super(MelTimeSaTask, self).__init__(nnet,
                                            frame_len=frame_len,
                                            frame_hop=frame_hop,
                                            window=window,
                                            center=center,
                                            round_pow_of_two=round_pow_of_two,
                                            stft_normalized=stft_normalized,
                                            permute=permute,
                                            num_spks=num_spks,
                                            weight=weight)
        mel = init_melfilter(None,
                             num_bins=num_bins,
                             sr=sr,
                             num_mels=num_mels,
                             fmax=fmax,
                             norm=mel_norm)
        self.mel = nn.Parameter(mel[..., None] * mel_scale, requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def transform(self, tensor):
        """
        Return mel spectrogram
        """
        if self.power_mag:
            tensor = tensor**2
        # N x F x T => N x M x T
        mel = tf.conv1d(tensor, self.mel, bias=None)
        if self.log:
            mel = th.log(1 + mel)
        return mel

    def objf(self, out, ref):
        """
        Computer MSE
        """
        loss = tf.mse_loss(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss


class ComplexMappingTask(SepTask):
    """
    Complex Spectral Mapping
    """

    def __init__(self, nnet, num_spks=2, weight=None, permute=True, objf="L1"):
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(ComplexMappingTask, self).__init__(nnet,
                                                 ctx=sa_ctx,
                                                 weight=weight)
        self.permute = permute
        self.num_spks = num_spks
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def _build_ref(self, wav):
        """
        Return real/imag part of the STFT
        """
        return self.ctx(wav, output="complex")

    def objf(self, out, ref):
        """
        Return loss for each mini-batch
        """
        out_mag = th.sqrt(out[0]**2 + out[1]**2)
        ref_mag = th.sqrt(ref[0]**2 + ref[1]**2)
        loss = self.objf_ptr(out[0], ref[0], reduction="none") + self.objf_ptr(
            out[1], ref[1], reduction="none") + self.objf_ptr(
                out_mag, ref_mag, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss

    def forward(self, egs, **kwargs):
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix = egs["mix"]
        # do separation or enhancement
        # out (real & imag parts): [Tensor, ...]
        out = self.nnet(mix)

        if isinstance(out, tuple):
            # F x T
            ref = self._build_ref(egs["ref"])
            # loss
            loss = self.objf(out, ref)
        else:
            if len(out) != len(egs["ref"]):
                raise RuntimeError(
                    f"Got {len(egs['ref'])} references but with {len(out)} outputs"
                )
            # for each reference
            ref = [self._build_ref(r) for r in egs["ref"]]
            loss = hybrid_objf(out,
                               ref,
                               self.objf,
                               weight=self.weight,
                               permute=self.permute,
                               permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}
