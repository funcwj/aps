#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
For Speech Separation and Enhancement task, using sse.py for abbreviation
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import List, Dict, Any, Tuple, Callable, Optional

from aps.task.base import Task
from aps.task.objf import permu_invarint_objf, multiple_objf, snr_objf, sisnr_objf, dpcl_objf
from aps.libs import ApsRegisters
from aps.transform.utils import STFT, mel_filter
from aps.const import EPSILON

__all__ = [
    "SisnrTask", "SnrTask", "WaTask", "LinearFreqSaTask", "LinearTimeSaTask",
    "MelFreqSaTask", "MelTimeSaTask", "ComplexMappingTask", "ComplexMaskingTask"
]


def hybrid_objf(out: List[Any],
                ref: List[Any],
                objf: Callable,
                transform: Optional[Callable] = None,
                weight: Optional[List[float]] = None,
                permute: bool = True,
                permu_num_spks: int = 2) -> th.Tensor:
    """
    Return hybrid loss (pair-wise, permutated or pair-wise + permutated)
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (function): function to compute single pair loss (per mini-batch)
        weight (list(float)): weight on each loss value
        permute (bool): use permutation invariant or not
        permu_num_spks (int): number of speakers when computing PIT
    """
    num_branch = len(out)
    if num_branch != len(ref):
        raise RuntimeError(
            f"Got {len(ref)} references but with {num_branch} outputs")

    if permute:
        # N
        loss = permu_invarint_objf(out[:permu_num_spks],
                                   ref[:permu_num_spks],
                                   objf,
                                   transform=transform)
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
        loss = multiple_objf(out, ref, objf, weight=weight, transform=transform)
    return loss


class SepTask(Task):
    """
    Base class for separation & enhancement task
    Args:
        nnet: network instance
        ctx: context network used for training if needed
        description: description string
        weight: weight on each output branch if needed
    """

    def __init__(self,
                 nnet: nn.Module,
                 ctx: Optional[nn.Module] = None,
                 description: str = "",
                 weight: Optional[str] = None) -> None:
        super(SepTask, self).__init__(nnet, ctx=ctx, description=description)
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
    Time domain task (to be inherited)
    Args:
        nnet: network instance
        num_spks: number of speakers (output branch in nnet)
        permute: use permutation invariant loss or not
        description: description string
        weight: weight on each output branch if needed
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 permute: bool = True,
                 description: str = "",
                 weight: Optional[str] = None) -> None:
        super(TimeDomainTask, self).__init__(nnet,
                                             weight=weight,
                                             description=description)
        self.num_spks = num_spks
        self.permute = permute  # use pit or not

    def forward(self, egs: Dict) -> Dict:
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
            out, ref = [out], [ref]
        loss = hybrid_objf(out,
                           ref,
                           self.objf,
                           weight=self.weight,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


@ApsRegisters.task.register("sse@sisnr")
class SisnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    Args:
        nnet: network instance
        num_spks: number of speakers (output branch in nnet)
        permute: use permutation invariant loss or not
        weight: weight on each output branch if needed
        zero_mean: force zero mean before computing sisnr loss
        non_nagetive: force non-nagetive value of sisnr
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 permute: bool = True,
                 weight: Optional[str] = None,
                 zero_mean: bool = True,
                 non_nagetive: bool = False) -> None:
        super(SisnrTask, self).__init__(
            nnet,
            num_spks=num_spks,
            permute=permute,
            weight=weight,
            description="Using SiSNR objective function for training")
        self.zero_mean = zero_mean
        self.non_nagetive = non_nagetive

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return negative SiSNR
        """
        return -sisnr_objf(
            out, ref, zero_mean=self.zero_mean, non_nagetive=self.non_nagetive)


@ApsRegisters.task.register("sse@snr")
class SnrTask(TimeDomainTask):
    """
    Time domain sisnr loss function
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 permute: bool = True,
                 weight: Optional[str] = None,
                 snr_max: float = -1,
                 non_nagetive: bool = False) -> None:
        super(SnrTask, self).__init__(
            nnet,
            num_spks=num_spks,
            permute=permute,
            weight=weight,
            description="Using SNR objective function for training")
        self.non_nagetive = non_nagetive
        self.snr_max = snr_max

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return negative SNR
        """
        return -snr_objf(
            out, ref, non_nagetive=self.non_nagetive, snr_max=self.snr_max)


@ApsRegisters.task.register("sse@wa")
class WaTask(TimeDomainTask):
    """
    Time domain waveform approximation loss function
    Args:
        nnet: network instance
        num_spks: number of speakers (output branch in nnet)
        objf: L1 or L2 loss
        permute: use permutation invariant loss or not
        weight: weight on each output branch if needed
    """

    def __init__(self,
                 nnet: nn.Module,
                 objf: str = "L1",
                 num_spks: int = 2,
                 permute: bool = True,
                 weight: Optional[str] = None) -> None:
        super(WaTask, self).__init__(
            nnet,
            num_spks=num_spks,
            permute=permute,
            weight=weight,
            description="Using L1/L2 loss on waveform for training")
        # L2 or L1 loss
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        L1 or L2
        """
        loss = self.objf_ptr(out, ref, reduction="none")
        return loss.sum(-1)


class FreqSaTask(SepTask):
    """
    Frequenct SA Task (to be inherited)
    Args:
        nnet: network instance
        phase_sensitive: using phase sensitive loss function
        truncated: truncated value (relative) of the reference
        num_spks: number of speakers (output branch in nnet)
        masking: if the network predicts TF-mask, set it true.
                 if the network predicts spectrogram, set it false
        permute: use permutation invariant loss or not
        description: description string for current task
        weight: weight on each output branch if needed
        dpcl_weight: weight of the DPCL loss if needed
    """

    def __init__(self,
                 nnet: nn.Module,
                 phase_sensitive: bool = False,
                 truncated: float = -1,
                 permute: bool = True,
                 masking: bool = True,
                 num_spks: int = 2,
                 description: str = "",
                 dpcl_weight: float = 0,
                 weight: Optional[str] = None) -> None:
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(FreqSaTask, self).__init__(nnet,
                                         ctx=sa_ctx,
                                         weight=weight,
                                         description=description)
        if not masking and truncated > 0:
            raise ValueError(
                "Conflict parameters: masksing = True while truncated > 0")
        self.phase_sensitive = phase_sensitive
        self.truncated = truncated
        self.permute = permute
        self.masking = masking
        self.num_spks = num_spks
        self.dpcl_weight = dpcl_weight

    def _ref_mag(self,
                 mix_mag: th.Tensor,
                 mix_pha: th.Tensor,
                 ref: th.Tensor,
                 psa: bool = False,
                 truncated: float = -1) -> th.Tensor:
        """
        Compute reference magnitude for SA
        """
        in_polar = self.ctx(ref, return_polar=True)
        ref_mag, ref_pha = in_polar[..., 0], in_polar[..., 1]
        # use phase-sensitive approximation
        if psa:
            # non-negative
            pha_dif = th.clamp(th.cos(ref_pha - mix_pha), min=0)
            ref_mag = ref_mag * pha_dif
        # truncated
        if truncated > 0:
            ref_mag = th.min(ref_mag, truncated * mix_mag)
        return ref_mag

    def forward(self, egs: Dict) -> Dict:
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix, ref = egs["mix"], egs["ref"]
        # out: Tensor or [Tensor, ...]
        mask = self.nnet(mix)

        # if multi-channel, use ch0 as reference
        in_polar = self.ctx(mix[:, 0] if mix.dim() == 3 else mix,
                            return_polar=True)
        mix_mag, mix_pha = in_polar[..., 0], in_polar[..., 1]

        if isinstance(mask, th.Tensor):
            mask, ref = [mask], [ref]
        # for each reference
        ref_psa = [
            self._ref_mag(mix_mag,
                          mix_pha,
                          r,
                          psa=self.phase_sensitive,
                          truncated=self.truncated) for r in ref
        ]
        if self.masking:
            out = [m * mix_mag for m in mask]
        else:
            out = mask
        loss = hybrid_objf(out,
                           ref_psa,
                           self.objf,
                           transform=self.transform,
                           weight=self.weight,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        # if have dpcl branch
        enable_dpcl = self.dpcl_weight > 0 and hasattr(self.nnet, "dpcl_embed")
        if enable_dpcl and len(ref) >= 2:
            raw_mag = [
                self._ref_mag(mix_mag, mix_pha, r, psa=False, truncated=-1)
                for r in ref
            ]
            # classes id: N x F x T
            classes = th.argmax(th.stack(raw_mag, -1), -1)
            # weights: N x F x T
            weights = mix_mag / th.sum(mix_mag, (-1, -2), keepdim=True)
            dpcl_loss = dpcl_objf(self.nnet.dpcl_embed(),
                                  classes,
                                  weights,
                                  num_spks=self.num_spks,
                                  whitened=False)
            loss = (1 - self.dpcl_weight) * loss + self.dpcl_weight * dpcl_loss
        return {"loss": th.mean(loss)}


@ApsRegisters.task.register("sse@freq_linear_sa")
class LinearFreqSaTask(FreqSaTask):
    """
    Frequency domain linear spectral approximation (MSA or tPSA) loss function
    Args:
        nnet: network instance
        phase_sensitive: using phase sensitive loss function
        truncated: truncated value (relative) of the reference
        num_spks: number of speakers (output branch in nnet)
        masking: if the network predicts TF-mask, set it true.
                 if the network predicts spectrogram, set it false
        permute: use permutation invariant loss or not
        objf: L1 or L2 distance
        weight: weight on each output branch if needed
        dpcl_weight: weight of the DPCL loss if needed
    """

    def __init__(self,
                 nnet: nn.Module,
                 phase_sensitive: bool = False,
                 truncated: float = -1,
                 permute: bool = True,
                 masking: bool = True,
                 dpcl_weight: float = 0,
                 num_spks: int = 2,
                 objf: str = "L2",
                 weight: Optional[str] = None) -> None:
        super(LinearFreqSaTask,
              self).__init__(nnet,
                             phase_sensitive=phase_sensitive,
                             truncated=truncated,
                             permute=permute,
                             masking=masking,
                             weight=weight,
                             dpcl_weight=dpcl_weight,
                             num_spks=num_spks,
                             description="Using spectral approximation "
                             "(MSA or tPSA) loss function")
        # L2 or L1 loss
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return loss for each mini-batch
        """
        # out, ref: N x F x T
        loss = self.objf_ptr(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss

    def transform(self, tensor: th.Tensor) -> th.Tensor:
        """
        Just return itself
        """
        return tensor


@ApsRegisters.task.register("sse@freq_mel_sa")
class MelFreqSaTask(FreqSaTask):
    """
    Frequency domain mel-spectrogram approximation
    Args:
        nnet: network instance
        phase_sensitive: using phase sensitive loss function
        truncated: truncated value (relative) of the reference
        num_spks: number of speakers (output branch in nnet)
        masking: if the network predicts TF-mask, set it true.
                 if the network predicts spectrogram, set it false
        permute: use permutation invariant loss or not
        weight: weight on each output branch if needed
        dpcl_weight: weight of the DPCL loss if needed
        ...: others are parameters for mel-spectrogram computation
    """

    def __init__(self,
                 nnet: nn.Module,
                 phase_sensitive: bool = False,
                 truncated: float = -1,
                 weight: Optional[str] = None,
                 dpcl_weight: float = 0,
                 permute: bool = True,
                 num_spks: int = 2,
                 masking: bool = True,
                 power_mag: bool = False,
                 num_bins: int = 257,
                 num_mels: int = 80,
                 mel_log: int = False,
                 mel_scale: int = 1,
                 mel_norm: bool = False,
                 sr: int = 16000,
                 fmax: int = 8000) -> None:
        super(MelFreqSaTask,
              self).__init__(nnet,
                             phase_sensitive=phase_sensitive,
                             truncated=truncated,
                             permute=permute,
                             masking=masking,
                             weight=weight,
                             dpcl_weight=dpcl_weight,
                             num_spks=num_spks,
                             description="Using L2 loss of the mel features")
        mel = mel_filter(None,
                         num_bins=num_bins,
                         sr=sr,
                         num_mels=num_mels,
                         fmax=fmax,
                         norm=mel_norm)
        self.mel = nn.Parameter(mel[..., None] * mel_scale, requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def transform(self, tensor: th.Tensor) -> th.Tensor:
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

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Computer MSE after mel-transform
        """
        loss = tf.mse_loss(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss


class TimeSaTask(SepTask):
    """
    Time domain spectral approximation Task. The network output time-domain signals,
    we transform them to frequency domain and then compute loss function
    Args:
        nnet: network instance
        frame_len: length of the frame (used in STFT)
        frame_hop: hop size between frames (used in STFT)
        window: window name (used in STFT)
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        stft_normalized: use normalized DFT kernel in STFT
        pre_emphasis: coefficient of preemphasis
        num_spks: number of speakers (output branch in nnet)
        weight: weight on each output branch if needed
        permute: use permutation invariant loss or not
        description: description string on current task
    """

    def __init__(self,
                 nnet: nn.Module,
                 frame_len: int = 512,
                 frame_hop: int = 256,
                 center: bool = False,
                 window: str = "sqrthann",
                 round_pow_of_two: bool = True,
                 stft_normalized: bool = False,
                 pre_emphasis: float = 0,
                 permute: bool = True,
                 weight: Optional[float] = None,
                 num_spks: int = 2,
                 description: str = "") -> None:
        # STFT context
        sa_ctx = STFT(frame_len,
                      frame_hop,
                      window=window,
                      center=center,
                      round_pow_of_two=round_pow_of_two,
                      normalized=stft_normalized)
        super(TimeSaTask, self).__init__(nnet,
                                         ctx=sa_ctx,
                                         weight=weight,
                                         description=description)
        self.permute = permute
        self.num_spks = num_spks
        self.pre_emphasis = pre_emphasis

    def _stft_mag(self, wav: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute STFT magnitude for SA loss
        """
        # for ASR (do pre-emphasis)
        if self.pre_emphasis > 0:
            wav[:, 1:] = wav[:, 1:] - self.pre_emphasis * wav[:, :-1]
        in_polar = self.ctx(wav, return_polar=True)
        return in_polar[..., 0]

    def forward(self, egs: Dict) -> Dict:
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix, ref = egs["mix"], egs["ref"]
        # spk: Tensor or [Tensor, ...]
        spk = self.nnet(mix)

        if isinstance(spk, th.Tensor):
            spk, ref = [spk], [ref]
        spk_mag = [self._stft_mag(s) for s in spk]
        ref_mag = [self._stft_mag(r) for r in ref]
        # post processing
        # out = [self.transform(s) for s in spk_mag]
        # ref = [self.transform(r) for r in ref_mag]
        loss = hybrid_objf(spk_mag,
                           ref_mag,
                           self.objf,
                           transform=self.transform,
                           weight=self.weight,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


@ApsRegisters.task.register("sse@time_linear_sa")
class LinearTimeSaTask(TimeSaTask):
    """
    Time domain linear spectral approximation loss function
    Args:
        nnet: network instance
        frame_len: length of the frame (used in STFT)
        frame_hop: hop size between frames (used in STFT)
        window: window name (used in STFT)
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        stft_normalized: use normalized DFT kernel in STFT
        num_spks: number of speakers (output branch in nnet)
        weight: weight on each output branch if needed
        permute: use permutation invariant loss or not
        objf: ise L1 or L2 distance
    """

    def __init__(self,
                 nnet: nn.Module,
                 frame_len: int = 512,
                 frame_hop: int = 256,
                 center: bool = False,
                 window: str = "sqrthann",
                 round_pow_of_two: bool = True,
                 stft_normalized: bool = False,
                 permute: bool = True,
                 weight: Optional[str] = None,
                 num_spks: int = 2,
                 objf: str = "L2") -> None:
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
                             weight=weight,
                             description="Using L1/L2 loss on magnitude "
                             "of the waveform")
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


@ApsRegisters.task.register("sse@time_mel_sa")
class MelTimeSaTask(TimeSaTask):
    """
    Time domain mel spectral approximation loss function
    Args:
        nnet: network instance
        frame_len: length of the frame (used in STFT)
        frame_hop: hop size between frames (used in STFT)
        window: window name (used in STFT)
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        stft_normalized: use normalized DFT kernel in STFT
        num_spks: number of speakers (output branch in nnet)
        weight: weight on each output branch if needed
        permute: use permutation invariant loss or not
        ...: others are parameters for mel-spectrogam extraction
    """

    def __init__(self,
                 nnet: nn.Module,
                 frame_len: int = 512,
                 frame_hop: int = 256,
                 window: str = "sqrthann",
                 center: bool = False,
                 round_pow_of_two: bool = True,
                 stft_normalized: bool = False,
                 permute: bool = True,
                 weight: Optional[str] = None,
                 num_spks: int = 2,
                 num_bins: int = 257,
                 num_mels: int = 80,
                 power_mag: bool = False,
                 mel_log: bool = False,
                 mel_scale: int = 1,
                 mel_norm: bool = False,
                 sr: int = 16000,
                 fmax: int = 7690) -> None:
        super(MelTimeSaTask,
              self).__init__(nnet,
                             frame_len=frame_len,
                             frame_hop=frame_hop,
                             window=window,
                             center=center,
                             round_pow_of_two=round_pow_of_two,
                             stft_normalized=stft_normalized,
                             permute=permute,
                             num_spks=num_spks,
                             weight=weight,
                             description="Using L2 loss on the mel "
                             "features of the waveform")
        mel = mel_filter(None,
                         num_bins=num_bins,
                         sr=sr,
                         num_mels=num_mels,
                         fmax=fmax,
                         norm=mel_norm)
        self.mel = nn.Parameter(mel[..., None] * mel_scale, requires_grad=False)
        self.log = mel_log
        self.power_mag = power_mag

    def transform(self, tensor: th.Tensor) -> th.Tensor:
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

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Computer MSE
        """
        loss = tf.mse_loss(out, ref, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss


@ApsRegisters.task.register("sse@complex_mapping")
class ComplexMappingTask(SepTask):
    """
    Frequency domain complex spectral mapping loss function
    Args:
        nnet: network instance
        num_spks: number of speakers (output branch in nnet)
        weight: weight on each output branch if needed
        permute: use permutation invariant loss or not
        objf: use L1 or L2 distance
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 weight: Optional[str] = None,
                 permute: bool = True,
                 objf: str = "L1",
                 add_magnitude_loss: bool = True) -> None:
        # STFT context
        sa_ctx = nnet.enh_transform.ctx("forward_stft")
        super(ComplexMappingTask, self).__init__(
            nnet,
            ctx=sa_ctx,
            weight=weight,
            description="Using complex mapping function for training")
        self.permute = permute
        self.num_spks = num_spks
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss
        self.add_magnitude_loss = add_magnitude_loss

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return loss for each mini-batch
        """
        real_loss = self.objf_ptr(out[..., 0], ref[..., 0], reduction="none")
        imag_loss = self.objf_ptr(out[..., 1], ref[..., 1], reduction="none")
        loss = real_loss + imag_loss
        if self.add_magnitude_loss:
            out_mag = th.sqrt(out[..., 0]**2 + out[..., 1]**2 + EPSILON)
            ref_mag = th.sqrt(ref[..., 0]**2 + ref[..., 1]**2 + EPSILON)
            loss += self.objf_ptr(out_mag, ref_mag, reduction="none")
        loss = th.sum(loss.mean(-1), -1)
        return loss

    def forward(self, egs: Dict) -> Dict:
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        mix, ref = egs["mix"], egs["ref"]
        # out: Tensor ([real, imag])
        out = self.nnet(mix)

        if isinstance(out, th.Tensor):
            out, ref = [out], [ref]
        # for each reference
        ref = [self.ctx(r, return_polar=False) for r in ref]
        loss = hybrid_objf(out,
                           ref,
                           self.objf,
                           weight=self.weight,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}


@ApsRegisters.task.register("sse@complex_masking")
class ComplexMaskingTask(ComplexMappingTask):
    """
    Frequency domain complex mask loss function
    Args:
        nnet: network instance
        num_spks: number of speakers (output branch in nnet)
        weight: weight on each output branch if needed
        permute: use permutation invariant loss or not
        objf: use L1 or L2 distance
        compress_param: cirm compression parameters
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 weight: Optional[str] = None,
                 permute: bool = True,
                 compress_param: Tuple[float] = [10, 0.1, -100],
                 compress_masks: bool = False,
                 objf: str = "L2") -> None:
        super(ComplexMaskingTask, self).__init__(nnet,
                                                 num_spks=num_spks,
                                                 weight=weight,
                                                 permute=permute,
                                                 objf=objf,
                                                 add_magnitude_loss=False)
        self.k, self.c, self.lower_bound = compress_param
        self.compress_masks = compress_masks

    def _compress_mask(self, mix_stft: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return compressed version of complex mask, ranges [-k, k]
        """
        ref_stft = self.ctx(ref, return_polar=False)
        denominator = th.sum(mix_stft**2, -1) + EPSILON
        real = (mix_stft[..., 0] * ref_stft[..., 0] +
                mix_stft[..., 1] * ref_stft[..., 1])
        imag = (mix_stft[..., 0] * ref_stft[..., 1] -
                mix_stft[..., 1] * ref_stft[..., 0])
        crm = th.stack([real, imag], -1) / denominator
        exp = th.exp(-self.c * th.clamp_min(crm, self.lower_bound))
        return self.k * (1 - exp) / (1 + exp)

    def _complex_tf_mask(self, mix_stft: th.Tensor,
                         mask: th.Tensor) -> th.Tensor:
        """
        Return TF masking result using the complex masks
        """
        real = (mix_stft[..., 0] * mask[..., 0] -
                mix_stft[..., 1] * mask[..., 1])
        imag = (mix_stft[..., 0] * mask[..., 1] +
                mix_stft[..., 1] * mask[..., 0])
        return th.stack([real, imag], -1)

    def forward(self, egs: Dict) -> Dict:
        """
        Return chunk-level loss
        egs contains:
            mix (Tensor): N x (C) x S
            ref (Tensor or [Tensor, ...]): N x S
        """
        ref = egs["ref"]
        # do separation or enhancement
        out = self.nnet(egs["mix"])

        if isinstance(out, th.Tensor):
            out, ref = [out], [ref]

        mix = self.ctx(egs["mix"], return_polar=False)
        # for each reference
        if self.compress_masks:
            ref = [self._compress_mask(mix, r) for r in ref]
        else:
            ref = [self.ctx(r, return_polar=False) for r in ref]
            out = [self._complex_tf_mask(mix, o) for o in out]
        loss = hybrid_objf(out,
                           ref,
                           self.objf,
                           weight=self.weight,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}
