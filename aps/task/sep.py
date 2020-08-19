#!/usr/bin/env python

# wujian@2020

import warnings
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from itertools import permutations

from aps.task.base import Task
from aps.transform.utils import STFT, init_melfilter
from aps.const import EPSILON

__all__ = [
    "SisnrTask", "SnrTask", "WaTask", "LinearFreqSaTask", "LinearTimeSaTask",
    "MelFreqSaTask", "MelTimeSaTask"
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


class TimeDomainTask(Task):
    """
    Time domain task (to be implemented)
    """
    def __init__(self,
                 nnet,
                 num_spks=2,
                 permute=True,
                 mode="max",
                 weight=None):
        super(TimeDomainTask, self).__init__(nnet, weight=weight)
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
            num_branch = len(out)
            if num_branch != len(ref):
                raise RuntimeError(
                    f"Got {len(ref)} references but with {num_branch} outputs")
            if self.permute:
                # P x N
                loss_mat = th.stack([
                    self._perm_objf(p, out, ref)
                    for p in permutations(range(self.num_spks))
                ])
                # NOTE: max or min
                if self.mode == "max":
                    loss, _ = th.max(loss_mat, dim=0)
                else:
                    loss, _ = th.min(loss_mat, dim=0)
                # add residual loss
                if num_branch > self.num_spks:
                    warnings.warn(
                        f"#Branch: {num_branch} > #Speaker: {self.num_spks}")
                    num_weight = num_branch - (self.num_spks - 1)
                    if self.weight is None:
                        self.weight = [1 / num_weight] * num_weight
                    if len(self.weight) != num_weight:
                        raise RuntimeError(
                            f"Missing weight ({self.weight}) for {num_branch} branch"
                        )
                    res_loss = [
                        self._objf(o, r) for o, r in zip(
                            out[self.num_spks:], ref[self.num_spks:])
                    ]
                    res_loss = sum(
                        [s * l for s, l in zip(self.weight[1:], res_loss)])
                    loss = self.weight[0] * loss + res_loss
            else:
                if self.weight is None:
                    self.weight = [1 / num_branch] * num_branch
                if len(self.weight) != num_branch:
                    raise RuntimeError(
                        f"Missing weight {self.weight} for {num_branch} branch"
                    )
                loss = [self._objf(o, r) for o, r in zip(out, ref)]
                loss = sum([s * l for s, l in zip(self.weight, loss)])
        if self.mode == "max":
            return -th.mean(loss), None
        else:
            return th.mean(loss), None


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
                                        mode="max",
                                        weight=weight)
        self.zero_mean = zero_mean
        self.non_nagetive = non_nagetive

    def _objf(self, out, ref):
        return sisnr(out,
                     ref,
                     zero_mean=self.zero_mean,
                     non_nagetive=self.non_nagetive)


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
                                      mode="max",
                                      weight=weight)
        self.non_nagetive = non_nagetive

    def _objf(self, out, ref):
        return snr(out, ref, non_nagetive=self.non_nagetive)


class WaTask(TimeDomainTask):
    """
    Time domain waveform approximation loss function
    """
    def __init__(self, nnet, objf="L1", num_spks=2, permute=True, weight=None):
        super(WaTask, self).__init__(nnet,
                                     num_spks=num_spks,
                                     permute=permute,
                                     mode="min",
                                     weight=weight)
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

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        raise NotImplementedError

    def _post(self, tensor):
        """
        Post processing on out/ref before calling _objf
        """
        raise NotImplementedError

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

    def _permu_objf(self, permute, out, ref):
        """
        SA computation in permutation mode
        """
        permu_loss = []
        # for one permutation
        for s, t in enumerate(permute):
            # N x F x T
            loss_mat = self._objf(out[s], ref[t], reduction="none")
            loss_utt = th.sum(loss_mat.mean(-1), -1)  # x N, per-frame
            permu_loss.append(loss_utt)
        # per-speaker
        return sum(permu_loss) / len(permute)

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
        _, _, T = mix_mag.shape

        if isinstance(mask, th.Tensor):
            # F x T
            ref = self._ref_mag(mix_mag, mix_pha, egs["ref"])
            # post processing
            out = self._post(mask * mix_mag if self.masking else mask)
            ref = self._post(ref)
            # loss
            loss = self._objf(out, ref, reduction="sum")
            # per-frame, per-minibatch
            loss = loss / (N * T)
        else:
            num_branch = len(mask)
            if num_branch != len(egs["ref"]):
                raise RuntimeError(
                    f"Got {len(egs['ref'])} references but with {num_branch} outputs"
                )
            # for each reference
            ref = [self._ref_mag(mix_mag, mix_pha, r) for r in egs["ref"]]
            if self.masking:
                out = [m * mix_mag for m in mask]
            else:
                out = mask
            ref = [self._post(r) for r in ref]
            out = [self._post(o) for o in out]
            if self.permute:
                # P x N
                permu_loss = th.stack([
                    self._permu_objf(p, out, ref)
                    for p in permutations(range(self.num_spks))
                ])
                # N, per-frame per-speaker
                min_val, _ = th.min(permu_loss, dim=0)
                # per-minibatch
                loss = th.mean(min_val)

                # add residual loss
                if num_branch > self.num_spks:
                    warnings.warn(
                        f"#Branch: {num_branch} > #Speaker: {self.num_spks}")
                    num_weight = num_branch - (self.num_spks - 1)
                    if self.weight is None:
                        self.weight = [1 / num_weight] * num_weight
                    if len(self.weight) != num_weight:
                        raise RuntimeError(
                            f"Missing weight ({self.weight}) for {num_branch} branch"
                        )
                    res_loss = [
                        # scale, per-frame, per-minibatch
                        self._objf(o, r, reduction="sum") / (N * T) for o, r in
                        zip(out[self.num_spks:], ref[self.num_spks:])
                    ]
                    res_loss = sum(
                        [s * l for s, l in zip(self.weight[1:], res_loss)])
                    # per-frame, per-minibatch, weight and sum
                    loss = self.weight[0] * loss + res_loss
            else:
                if self.weight is None:
                    self.weight = [1 / num_branch] * num_branch
                if len(self.weight) != num_branch:
                    raise RuntimeError(
                        f"Missing weight {self.weight} for {num_branch} branch"
                    )
                loss = [
                    # per-frame, per-minibatch
                    self._objf(o, r, reduction="sum") / (N * T)
                    for o, r in zip(out, ref)
                ]
                # weight and sum
                loss = sum([s * l for s, l in zip(self.weight, loss)])
        return loss, None


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
        self.objf = objf

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        if self.objf == "L1":
            loss = tf.l1_loss(out, ref, reduction=reduction)
        else:
            loss = tf.mse_loss(out, ref, reduction=reduction)
        return loss

    def _post(self, tensor):
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
        self.mel = nn.Parameter(mel[..., None] * mel_scale,
                                requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def _post(self, tensor):
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

    def _objf(self, out, ref, reduction="none"):
        """
        Computer MSE after mel-transform
        """
        return tf.mse_loss(out, ref, reduction=reduction)


class TimeSaTask(Task):
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

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        raise NotImplementedError

    def _post(self, tensor):
        """
        Post processing on tensor
        """
        raise NotImplementedError

    def _stft_mag(self, wav):
        """
        Compute STFT magnitude for SA loss
        """
        # for ASR (do pre-emphasis)
        if self.pre_emphasis > 0:
            wav[:, 1:] = wav[:, 1:] - self.pre_emphasis * wav[:, :-1]
        mag, _ = self.ctx(wav, output="polar")
        return mag

    def _permu_sa(self, permute, out, ref):
        """
        SA computation in permutation mode
        """
        permu_loss = []
        # for one permutation
        for s, t in enumerate(permute):
            # N x F x T
            loss_mat = self._objf(out[s], ref[t], reduction="none")
            loss_utt = th.sum(loss_mat.mean(-1), -1)  # x N, per-frame
            permu_loss.append(loss_utt)
        return sum(permu_loss) / len(permute)

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
        # spk: Tensor or [Tensor, ...]
        spk = self.nnet(mix)

        if isinstance(spk, th.Tensor):
            # F x T
            spk_mag = self._stft_mag(spk)
            ref_mag = self._stft_mag(egs["ref"])
            # post process
            spk_mag = self._post(spk_mag)
            ref_mag = self._post(ref_mag)
            # loss
            loss = self._objf(spk_mag, ref_mag, reduction="sum")
            # per-frame loss
            loss = loss / (N * ref_mag.shape[-1])
        else:
            num_branch = len(spk)
            if num_branch != len(egs["ref"]):
                raise RuntimeError(
                    f"Got {len(egs['ref'])} reference but with {num_branch} outputs"
                )
            spk_mag = [self._stft_mag(s) for s in spk]
            # for each reference
            ref_mag = [self._stft_mag(r) for r in egs["ref"]]
            # post
            spk_mag = [self._post(s) for s in spk_mag]
            ref_mag = [self._post(r) for r in ref_mag]

            _, _, T = spk_mag[0].shape

            if self.permute:
                # P x N
                permu_loss = th.stack([
                    self._permu_sa(p, spk_mag, ref_mag)
                    for p in permutations(range(self.num_spks))
                ])
                # N
                min_val, _ = th.min(permu_loss, dim=0)
                loss = th.mean(min_val)

                # add residual loss
                if num_branch > self.num_spks:
                    warnings.warn(
                        f"#Branch: {num_branch} > #Speaker: {self.num_spks}")
                    num_weight = num_branch - (self.num_spks - 1)
                    if self.weight is None:
                        self.weight = [1 / num_weight] * num_weight
                    if len(self.weight) != num_weight:
                        raise RuntimeError(
                            f"Missing weight ({self.weight}) for {num_branch} branch"
                        )
                    res_loss = [
                        # scale, per-frame, per-minibatch
                        self._objf(s, r, reduction="sum") / (N * T) for s, r in
                        zip(spk_mag[self.num_spks:], ref_mag[self.num_spks:])
                    ]
                    res_loss = sum(
                        [s * l for s, l in zip(self.weight[1:], res_loss)])
                    # per-frame, per-minibatch, weight and sum
                    loss = self.weight[0] * loss + res_loss
            else:
                if self.weight is None:
                    self.weight = [1 / num_branch] * num_branch
                if len(self.weight) != num_branch:
                    raise RuntimeError(
                        f"Missing weight {self.weight} for {num_branch} branch"
                    )
                loss = [
                    # per-frame, per-minibatch
                    self._objf(s, r, reduction="sum")
                    for s, r in zip(spk_mag, ref_mag)
                ]
                # weight and sum
                loss = sum([s * l for s, l in zip(self.weight, loss)])
        return loss, None


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
        self.objf = objf

    def _objf(self, out, ref, reduction="none"):
        """
        Return loss for each mini-batch
        """
        if self.objf == "L1":
            loss = tf.l1_loss(out, ref, reduction=reduction)
        else:
            loss = tf.mse_loss(out, ref, reduction=reduction)
        return loss

    def _post(self, tensor):
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
        self.mel = nn.Parameter(mel[..., None] * mel_scale,
                                requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def _post(self, tensor):
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

    def _objf(self, out, ref, reduction="none"):
        """
        Computer MSE
        """
        return tf.mse_loss(out, ref, reduction=reduction)