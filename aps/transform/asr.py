#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Feature transform for ASR
Notations:
    N: batch size
    C: number of channels
    T: number of frames
    F: number of FFT bins
    S: number of samples in utts
"""
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union
from aps.transform.utils import STFT, init_melfilter, speed_perturb_filter
from aps.transform.augment import tf_mask, perturb_speed
from aps.const import EPSILON
from aps.libs import ApsRegisters
from aps.cplx import ComplexTensor

from scipy.fftpack import dct
from kaldi_python_io.functional import read_kaldi_mat

AsrReturnType = Union[th.Tensor, Optional[th.Tensor]]


def detect_nan(feature: th.Tensor) -> th.Tensor:
    """
    Check if nan exists
    """
    num_nans = th.sum(th.isnan(feature))
    if num_nans:
        raise ValueError(f"Detect {num_nans} NANs in feature matrices, " +
                         f"shape = {feature.shape}...")
    return feature


class SpeedPerturbTransform(nn.Module):
    """
    Transform layer for speed perturb
    Args:
        sr: sample rate of source signal
        perturb: speed perturb factors
    """

    def __init__(self, sr: int = 16000, perturb: str = "0.9,1.0,1.1") -> None:
        super(SpeedPerturbTransform, self).__init__()
        self.sr = sr
        self.factor_str = perturb
        dst_sr = [int(factor * sr) for factor in map(float, perturb.split(","))]
        if not len(dst_sr):
            raise ValueError("No perturb options for doing speed perturb")
        # N x dst_sr x src_sr x K
        self.weights = nn.ParameterList([
            nn.Parameter(speed_perturb_filter(sr, fs), requires_grad=False)
            for fs in dst_sr
            if fs != sr
        ])
        self.last_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sr={self.sr}, factor={self.factor_str})"

    def output_length(self,
                      inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Compute output length after speed perturb
        """
        if self.last_weight is None:
            return inp_len
        if inp_len is None:
            return None
        dst_sr, src_sr, _ = self.last_weight.shape
        return (inp_len // src_sr) * dst_sr

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x ... x S
        Return:
            wav (Tensor): output signal, N x ... x S
        """
        self.last_weight = None
        if not self.training:
            return wav
        if wav.dim() != 2:
            raise RuntimeError(f"Now only supports 2D tensor, got {wav.dim()}")
        # 1.0, do not apply speed perturb
        choice = th.randint(0, len(self.weights) + 1, (1,)).item()
        if choice == len(self.weights):
            return wav
        else:
            self.last_weight = self.weights[choice]
            return perturb_speed(wav, self.last_weight)


class TFTransposeTransform(nn.Module):
    """
    Swap time/frequency axis
    Args:
        axis1, axis2: axis pairs to be transposed
    """

    def __init__(self, axis1: int = -1, axis2: int = -2) -> None:
        super(TFTransposeTransform, self).__init__()
        self.axis1 = axis1
        self.axis2 = axis2

    def extra_repr(self) -> str:
        return f"axis1={self.axis1}, axis2={self.axis2}"

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Args:
            tensor (Tensor): input signal, N x ... x F x T
        Return:
            tensor (Tensor): output signal, N x ... x T x F
        """
        return tensor.transpose(-1, -2)


class PreEmphasisTransform(nn.Module):
    """
    Do utterance level preemphasis
    Args:
        pre_emphasis: preemphasis factor
    """

    def __init__(self, pre_emphasis: float = 0) -> None:
        super(PreEmphasisTransform, self).__init__()
        self.pre_emphasis = pre_emphasis

    def extra_repr(self) -> str:
        return f"pre_emphasis={self.pre_emphasis}"

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x (C) x S
        Return:
            wav (Tensor): output signal, N x (C) x S
        """
        if self.pre_emphasis > 0:
            wav[..., 1:] = wav[..., 1:] - self.pre_emphasis * wav[..., :-1]
        return wav


class SpectrogramTransform(STFT):
    """
    Compute spectrogram as a layer
    Args:
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        pre_emphasis: factor of preemphasis
        normalized: use normalized DFT kernel
        onesided: output onesided STFT
        mode: "kaldi"|"librosa", slight difference on applying window function
        power: return power spectrogram or not
    """

    def __init__(self,
                 frame_len: int,
                 frame_hop: int,
                 center: bool = False,
                 window: str = "hamm",
                 round_pow_of_two: bool = True,
                 normalized: bool = False,
                 pre_emphasis: float = 0.97,
                 onesided: bool = True,
                 mode: str = "librosa",
                 use_power: bool = False) -> None:
        super(SpectrogramTransform,
              self).__init__(frame_len,
                             frame_hop,
                             center=center,
                             window=window,
                             round_pow_of_two=round_pow_of_two,
                             pre_emphasis=pre_emphasis,
                             normalized=normalized,
                             onesided=onesided,
                             mode=mode)
        self.use_power = use_power

    def dim(self):
        return self.num_bins

    def len(self, xlen: th.Tensor) -> th.Tensor:
        return self.num_frames(xlen)

    def extra_repr(self) -> str:
        return self.expr + f", use_power={self.use_power}"

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x (C) x S
        Return:
            mag (Tensor): magnitude, N x (C) x F x T
        """
        # N x (C) x F x T
        mag, _ = super().forward(wav)
        if self.use_power:
            mag = mag**2
        return mag


class AbsTransform(nn.Module):
    """
    Absolute transform
    Args:
        eps: small floor value to avoid NAN when backward
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super(AbsTransform, self).__init__()
        self.eps = eps

    def extra_repr(self) -> str:
        return f"eps={self.eps:.3e}"

    def forward(self, tensor: Union[th.Tensor, ComplexTensor]) -> th.Tensor:
        """
        Args:
            tensor (Tensor or ComplexTensor): N x T x F
        Return:
            tensor (Tensor): N x T x F
        """
        if not isinstance(tensor, th.Tensor):
            tensor = tensor + self.eps
        return tensor.abs()


class PowerTransform(nn.Module):
    """
    Power transform
    """

    def __init__(self) -> None:
        super(PowerTransform, self).__init__()

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Args:
            tensor (Tensor): N x T x F
        Return:
            tensor (Tensor): N x T x F
        """
        return tensor**2


class MelTransform(nn.Module):
    """
    Mel tranform as a layer
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        sr: sample rate of souce signal
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        requires_grad: make it trainable or not
    """

    def __init__(self,
                 frame_len: int,
                 round_pow_of_two: bool = True,
                 sr: int = 16000,
                 num_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None,
                 requires_grad: bool = False) -> None:
        super(MelTransform, self).__init__()
        # num_mels x (N // 2 + 1)
        filters = init_melfilter(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels,
                                 fmax=fmax,
                                 fmin=fmin)
        self.num_mels, self.num_bins = filters.shape
        self.filters = nn.Parameter(filters, requires_grad=requires_grad)
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax

    def dim(self) -> int:
        return self.num_mels

    def extra_repr(self) -> str:
        return ("fmin={0}, fmax={1}, mel_filter={2[0]}x{2[1]}".format(
            self.fmin, self.fmax, self.filters.shape))

    def forward(self, linear: th.Tensor) -> th.Tensor:
        """
        Args:
            linear (Tensor): linear spectrogram, N x (C) x T x F
        Return:
            fbank (Tensor): mel-fbank feature, N x (C) x T x B
        """
        if linear.dim() not in [3, 4]:
            raise RuntimeError("MelTransform expect 3/4D tensor, " +
                               f"but got {linear.dim()} instead")
        # N x T x F => N x T x M
        fbank = F.linear(linear, self.filters, bias=None)
        return fbank


class LogTransform(nn.Module):
    """
    Transform linear domain to log domain
    Args:
        eps: floor value to avoid nagative values
        lower_bound: lower bound value
    """

    def __init__(self, eps: float = 1e-5, lower_bound: float = 0) -> None:
        super(LogTransform, self).__init__()
        self.eps = eps
        self.lower_bound = lower_bound

    def dim_scale(self) -> int:
        return 1

    def extra_repr(self) -> str:
        return f"eps={self.eps:.3e}, lower_bound={self.lower_bound}"

    def forward(self, linear: th.Tensor) -> th.Tensor:
        """
        Args:
            linear (Tensor): linear feature, N x (C) x T x F
        Return:
            logf (Tensor): log features, N x (C) x T x F
        """
        if self.lower_bound > 0:
            linear = self.lower_bound + linear
        else:
            linear = th.clamp(linear, min=self.eps)
        return th.log(linear)


class DiscreteCosineTransform(nn.Module):
    """
    DCT as a layer (for mfcc features)
    Args:
        num_ceps: number of the cepstrum coefficients
        num_mels: number of mel bands
        lifter: lifter factor
    """

    def __init__(self,
                 num_ceps: int = 13,
                 num_mels: int = 40,
                 lifter: float = 0) -> None:
        super(DiscreteCosineTransform, self).__init__()
        self.lifter = lifter
        self.num_ceps = num_ceps

        # num_ceps x num_mels
        self.dct = nn.Parameter(th.tensor(dct(th.eye(num_mels).numpy(),
                                              norm="ortho")[:num_ceps],
                                          dtype=th.float32),
                                requires_grad=False)
        if lifter > 0:
            cepstral_lifter = 1 + lifter * 0.5 * th.sin(
                math.pi * th.arange(1, 1 + num_ceps) / lifter)
            self.cepstral_lifter = nn.Parameter(cepstral_lifter,
                                                requires_grad=False)
        else:
            self.cepstral_lifter = None

    def dim(self) -> int:
        return self.num_ceps

    def extra_repr(self) -> str:
        return "cepstral_lifter={0}, dct={1[0]}x{1[1]}".format(
            self.lifter, self.dct.shape)

    def forward(self, log_mel: th.Tensor) -> th.Tensor:
        """
        Args:
            log_mel (Tensor): log mel-fbank, N x (C) x T x B
        Return:
            mfcc (Tensor): mfcc feature, N x (C) x T x P
        """
        mfcc = F.linear(log_mel, self.dct, bias=None)
        if self.cepstral_lifter is not None:
            mfcc = mfcc * self.cepstral_lifter
        return mfcc


class CmvnTransform(nn.Module):
    """
    Utterance & Global level mean & variance normalization
    Args:
        norm_mean: normalize mean or not
        norm_var: normalize var or not
        per_band: do cmvn perband or not
        gcmvn: path of the glabal cmvn statistics
        eps: small value to avoid NAN
    """

    def __init__(self,
                 norm_mean: bool = True,
                 norm_var: bool = True,
                 per_band: bool = True,
                 gcmvn: str = "",
                 eps: float = 1e-5) -> None:
        super(CmvnTransform, self).__init__()
        self.gmean, self.gstd = None, None
        if gcmvn:
            # in Kaldi format
            if gcmvn[-4:] == ".ark":
                cmvn = read_kaldi_mat(gcmvn)
                N = cmvn[0, -1]
                mean = th.tensor(cmvn[0, :-1] / N, dtype=th.float32)
                std = th.tensor(cmvn[1, :-1] / N - mean**2,
                                dtype=th.float32)**0.5
            else:
                stats = th.load(gcmvn)
                mean, std = stats[0], stats[1]
            self.gmean = nn.Parameter(mean, requires_grad=False)
            self.gstd = nn.Parameter(std, requires_grad=False)
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.per_band = per_band
        self.gcmvn = gcmvn
        self.eps = eps

    def extra_repr(self) -> str:
        return (
            f"norm_mean={self.norm_mean}, norm_var={self.norm_var}, per_band={self.per_band}"
            + f", gcmvn_stats={self.gcmvn}, eps={self.eps:.3e}")

    def dim_scale(self) -> int:
        return 1

    def forward(self, feats: th.Tensor) -> th.Tensor:
        """
        Args:
            feats (Tensor): feature before normalization, N x (C) x T x F
        Return:
            feats (Tensor): normalized feature, N x (C) x T x F
        """
        if not self.norm_mean and not self.norm_var:
            return feats
        if self.gmean is not None:
            if self.norm_mean:
                feats = feats - self.gmean
            if self.norm_var:
                feats = feats / self.gstd
        else:
            axis = -2 if self.per_band else (-1, -2)
            if self.norm_mean:
                feats = feats - th.mean(feats, axis, keepdim=True)
            if self.norm_var:
                if self.norm_mean:
                    var = th.mean(feats**2, axis, keepdim=True)
                else:
                    var = th.var(feats, axis, unbiased=False, keepdim=True)
                feats = feats / th.sqrt(var + self.eps)
        return feats


class SpecAugTransform(nn.Module):
    """
    Spectra data augmentation
    Args:
        p: probability to do spec-augment
        p_time: p in SpecAugment paper
        mask_zero: use zero value or mean in the masked region
        num_freq_masks|num_time_masks: m_F, m_T in the SpecAugment paper
        max_bands|max_frame: F, T in the SpecAugment paper
    """

    def __init__(self,
                 p: float = 0.5,
                 p_time: float = 1.0,
                 max_bands: int = 30,
                 max_frame: int = 40,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 2,
                 mask_zero: bool = True) -> None:
        super(SpecAugTransform, self).__init__()
        self.fnum, self.tnum = num_freq_masks, num_time_masks
        self.mask_zero = mask_zero
        self.F, self.T = max_bands, max_frame
        # prob to do spec-augment
        self.p = p
        # max portion constraint on time axis
        self.p_time = p_time

    def extra_repr(self) -> str:
        return (
            f"max_bands={self.F}, max_frame={self.T}, " +
            f"p={self.p}, p_time={self.p_time}, mask_zero={self.mask_zero}, "
            f"num_freq_masks={self.fnum}, num_time_masks={self.tnum}")

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): original features, N x (C) x T x F
        Return:
            y (Tensor): augmented features
        """
        if self.training and th.rand(1).item() < self.p:
            if x.dim() == 4:
                N, _, T, F = x.shape
            else:
                N, T, F = x.shape
            # N x T x F
            mask = tf_mask(N, (T, F),
                           p=self.p_time,
                           max_bands=self.F,
                           max_frame=self.T,
                           num_freq_masks=self.fnum,
                           num_time_masks=self.tnum,
                           device=x.device)
            if x.dim() == 4:
                # N x 1 x T x F
                mask = mask.unsqueeze(1)
            if self.mask_zero:
                x = x * mask
            else:
                x = th.masked_fill(x, mask == 0, x.mean())
        return x


class SpliceTransform(nn.Module):
    """
    Do feature splicing as well as downsampling if needed
    Args:
        lctx: left context
        rctx: right contex
        subsampling_factor: subsampling factor
    """

    def __init__(self,
                 lctx: int = 0,
                 rctx: int = 0,
                 subsampling_factor: int = 1) -> None:
        super(SpliceTransform, self).__init__()
        self.subsampling_factor = subsampling_factor
        self.lctx = max(lctx, 0)
        self.rctx = max(rctx, 0)

    def extra_repr(self) -> str:
        return (f"context=({self.lctx}, {self.rctx}), " +
                f"subsampling_factor={self.subsampling_factor}")

    def dim_scale(self) -> int:
        return (1 + self.rctx + self.lctx)

    def forward(self, feats: th.Tensor) -> th.Tensor:
        """
        args:
            feats (Tensor): original feature, N x ... x Ti x F
        return:
            slice (Tensor): spliced feature, N x ... x To x FD
        """
        T = feats.shape[-2]
        T = T - T % self.subsampling_factor
        if self.lctx + self.rctx != 0:
            ctx = []
            for c in range(-self.lctx, self.rctx + 1):
                idx = th.arange(c, c + T, device=feats.device, dtype=th.int64)
                idx = th.clamp(idx, min=0, max=T - 1)
                # N x ... x T x F
                ctx.append(th.index_select(feats, -2, idx))
            # N x ... x T x FD
            feats = th.cat(ctx, -1)
        if self.subsampling_factor != 1:
            feats = feats[..., ::self.subsampling_factor, :]
        return feats


class DeltaTransform(nn.Module):
    """
    Add delta features
    Args:
        ctx: context size
        order: delta order
    """

    def __init__(self, ctx: int = 2, order: int = 2) -> None:
        super(DeltaTransform, self).__init__()
        self.ctx = ctx
        self.order = order

    def extra_repr(self) -> str:
        return f"context={self.ctx}, order={self.order}"

    def dim_scale(self) -> int:
        return self.order

    def _add_delta(self, x: th.Tensor) -> th.Tensor:
        dx = th.zeros_like(x)
        for i in range(1, self.ctx + 1):
            dx[..., :-i, :] += i * x[..., i:, :]
            dx[..., i:, :] += -i * x[..., :-i, :]
            dx[..., -i:, :] += i * x[..., -1:, :]
            dx[..., :i, :] += -i * x[..., :1, :]
        dx = dx / (2 * sum(i**2 for i in range(1, self.ctx + 1)))
        return dx

    def forward(self, feats: th.Tensor) -> th.Tensor:
        """
        args:
            feats (Tensor): original feature, N x (C) x T x F
        return:
            delta (Tensor): delta feature, N x (C) x T x FD
        """
        delta = [feats]
        for _ in range(self.order):
            delta.append(self._add_delta(delta[-1]))
        # N x ... x T x FD
        return th.cat(delta, -1)


@ApsRegisters.transform.register("asr")
class FeatureTransform(nn.Module):
    """
    Feature transform for ASR tasks
        - SpeedPerturbTransform
        - PreEmphasisTransform
        - SpectrogramTransform
        - TFTransposeTransform
        - PowerTransform
        - MelTransform
        - AbsTransform
        - LogTransform
        - DiscreteCosineTransform
        - CmvnTransform
        - SpecAugTransform
        - SpliceTransform
        - DeltaTransform

    Args:
        feats: string that shows the way to extract features
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        stft_normalized: use normalized DFT kernel
        stft_mode: "kaldi"|"librosa", slight difference on windowing
        pre_emphasis: factor of preemphasis
        use_power: use power spectrogram or not
        sr: sample rate of the audio
        speed_perturb: speed perturb factors (perturb)
        log_lower_bound: lower_bound when we apply log (log)
        num_mels: number of the mel bands (for fbank|mfcc)
        num_ceps: number of the cepstrum coefficients (mfcc)
        min_freq|max_freq: frequency boundry
        lifter: lifter factor (mfcc)
        aug_prob: probability to do spec-augment
        aug_maxp_time: p in SpecAugment paper
        aug_mask_zero: use zero value or mean in the masked region
        num_aug_bands|num_aug_frame: m_F, m_T in the SpecAugment paper
        aug_max_bands|aug_max_frame: F, T in the SpecAugment paper
        norm_mean|norm_var: normalize mean/var or not (cmvn)
        norm_per_band: do cmvn per-band or not (cmvn)
        gcmvn: global cmvn statistics (cmvn)
        subsampling_factor: subsampling factor
        lctx|rctx: left/right context for splicing (splice)
        delta_ctx|delta_order: context|order used in delta feature (delta)
        requires_grad: make mel matrice trainable
        eps: floor number
    """

    def __init__(self,
                 feats: str = "fbank-log-cmvn",
                 frame_len: int = 400,
                 frame_hop: int = 160,
                 window: str = "hamm",
                 center: bool = False,
                 round_pow_of_two: bool = True,
                 stft_normalized: bool = False,
                 stft_mode: str = "librosa",
                 pre_emphasis: float = 0.97,
                 use_power: bool = False,
                 sr: int = 16000,
                 speed_perturb: str = "0.9,1.0,1.1",
                 log_lower_bound: float = 0,
                 num_mels: int = 80,
                 num_ceps: int = 13,
                 min_freq: int = 0,
                 max_freq: Optional[int] = None,
                 lifter: float = 0,
                 aug_prob: float = 0,
                 aug_maxp_time: float = 1.0,
                 aug_max_bands: int = 30,
                 aug_max_frame: int = 40,
                 aug_mask_zero: bool = True,
                 num_aug_bands: int = 1,
                 num_aug_frame: int = 1,
                 norm_mean: bool = True,
                 norm_var: bool = True,
                 norm_per_band: bool = True,
                 gcmvn: str = "",
                 subsampling_factor: int = 1,
                 lctx: int = 1,
                 rctx: int = 1,
                 delta_ctx: int = 2,
                 delta_order: int = 2,
                 requires_grad: bool = False,
                 eps: float = EPSILON) -> None:
        super(FeatureTransform, self).__init__()
        if not feats:
            raise ValueError("FeatureTransform: \'feats\' can not be empty")
        feat_tokens = feats.split("-")
        transform = []
        feats_dim = 0
        stft_kwargs = {
            "mode": stft_mode,
            "window": window,
            "center": center,
            "use_power": use_power,
            "pre_emphasis": pre_emphasis,
            "normalized": stft_normalized,
            "round_pow_of_two": round_pow_of_two
        }
        mel_kwargs = {
            "round_pow_of_two": round_pow_of_two,
            "sr": sr,
            "fmin": min_freq,
            "fmax": max_freq,
            "num_mels": num_mels,
            "requires_grad": requires_grad
        }
        self.spectra_index = -1
        self.perturb_index = -1
        for idx, tok in enumerate(feat_tokens):
            if tok == "perturb":
                transform.append(
                    SpeedPerturbTransform(sr=sr, perturb=speed_perturb))
                self.perturb_index = idx
            elif tok == "emph":
                transform.append(
                    PreEmphasisTransform(pre_emphasis=pre_emphasis))
            elif tok == "spectrogram":
                spectrogram = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    TFTransposeTransform(),
                ]
                transform += spectrogram
                feats_dim = transform[-2].dim()
                self.spectra_index = idx
            elif tok == "trans":
                transform.append(TFTransposeTransform())
            elif tok == "power":
                transform.append(PowerTransform())
            elif tok == "fbank":
                fbank = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    TFTransposeTransform(),
                    MelTransform(frame_len, **mel_kwargs)
                ]
                self.spectra_index = idx
                transform += fbank
                feats_dim = transform[-1].dim()
            elif tok == "mfcc":
                mfcc = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    TFTransposeTransform(),
                    MelTransform(frame_len, **mel_kwargs),
                    LogTransform(eps=eps),
                    DiscreteCosineTransform(num_ceps=num_ceps,
                                            num_mels=num_mels,
                                            lifter=lifter)
                ]
                self.spec_index = idx
                transform += mfcc
                feats_dim = transform[-1].dim()
            elif tok == "mel":
                transform.append(MelTransform(frame_len, **mel_kwargs))
                feats_dim = transform[-1].dim()
            elif tok == "log":
                transform.append(
                    LogTransform(eps=eps, lower_bound=log_lower_bound))
            elif tok == "abs":
                transform.append(AbsTransform(eps=eps))
            elif tok == "dct":
                transform.append(
                    DiscreteCosineTransform(num_ceps=num_ceps,
                                            num_mels=num_mels,
                                            lifter=lifter))
                feats_dim = transform[-1].dim()
            elif tok == "cmvn":
                transform.append(
                    CmvnTransform(norm_mean=norm_mean,
                                  norm_var=norm_var,
                                  per_band=norm_per_band,
                                  gcmvn=gcmvn,
                                  eps=eps))
            elif tok == "aug":
                transform.append(
                    SpecAugTransform(p=aug_prob,
                                     p_time=aug_maxp_time,
                                     max_bands=aug_max_bands,
                                     max_frame=aug_max_frame,
                                     mask_zero=aug_mask_zero,
                                     num_freq_masks=num_aug_bands,
                                     num_time_masks=num_aug_frame))
            elif tok == "splice":
                transform.append(
                    SpliceTransform(lctx=lctx,
                                    rctx=rctx,
                                    subsampling_factor=subsampling_factor))
                feats_dim *= (1 + lctx + rctx)
            elif tok == "delta":
                transform.append(
                    DeltaTransform(ctx=delta_ctx, order=delta_order))
                feats_dim *= (1 + delta_order)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        self.transform = nn.Sequential(*transform)
        self.feats_dim = feats_dim
        self.subsampling_factor = subsampling_factor

    def num_frames(self, wav_len: th.Tensor) -> th.Tensor:
        """
        Work out number of frames
        """
        if wav_len is None:
            return None
        if self.spectra_index == -1:
            raise RuntimeError("No SpectrogramTransform layer is found, "
                               "can not work out number of the frames")
        if self.perturb_index != -1:
            wav_len = self.transform[self.perturb_index].output_length(wav_len)
        num_frames = self.transform[self.spectra_index].len(wav_len)
        return num_frames // self.subsampling_factor

    def forward(self, wav_pad: th.Tensor,
                wav_len: Optional[th.Tensor]) -> AsrReturnType:
        """
        Args:
            wav_pad (Tensor): raw waveform: N x C x S or N x S
            wav_len (Tensor or None): N or None
        Return:
            feats (Tensor): acoustic features: N x C x T x ...
            num_frames (Tensor or None): number of frames
        """
        feats = self.transform(wav_pad)
        return detect_nan(feats), self.num_frames(wav_len)
