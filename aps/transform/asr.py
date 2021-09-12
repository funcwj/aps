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
import warnings

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Union, Tuple
from aps.transform.utils import STFT, mel_filter, splice_feature, speed_perturb_filter
from aps.transform.augment import tf_mask, perturb_speed
from aps.const import EPSILON, MAX_INT16
from aps.libs import ApsRegisters

from scipy.fftpack import dct as scipy_dct
from kaldi_python_io.functional import read_kaldi_mat

AsrReturnType = Union[th.Tensor, Optional[th.Tensor]]


def check_valid(feature: th.Tensor,
                num_frames: Optional[th.Tensor]) -> Tuple[th.Tensor]:
    """
    Check NAN and valid of the tensor
    Args:
        feature: N x (C) x T x F
        num_frames: N or None
    """
    num_nans = th.sum(th.isnan(feature))
    shape = feature.shape
    if num_nans:
        raise ValueError(f"Detect {num_nans} NANs in feature matrices, " +
                         f"shape = {shape}...")
    if num_frames is not None:
        max_frames = num_frames.max().item()
        if feature.shape[-2] < max_frames:
            raise RuntimeError(f"feats shape: {shape[-2]} x {shape[-1]}, " +
                               f"num_frames = {num_frames.tolist()}")
        if feature.shape[-2] > max_frames:
            feature = feature[..., :max_frames, :]
    return feature, num_frames


class RescaleTransform(nn.Module):
    """
    Rescale audio samples (e.g., [-1, 1] to MAX_INT16 scale)

    By define this layer, we can avoid using "audio_norm" parameters in dataloader
    and make it easy to control during training and evaluation (decoding)

    Args:
        rescale: rescale number, MAX_INT16 by default
    """

    def __init__(self, rescale: float = MAX_INT16 * 1.0) -> None:
        super(RescaleTransform, self).__init__()
        self.rescale = rescale

    def extra_repr(self) -> str:
        return f"rescale={self.rescale}"

    def exportable(self) -> bool:
        return False

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x (C) x S
        Return:
            wav (Tensor): output signal, N x (C) x S
        """
        return th.round(wav * self.rescale)


class PreEmphasisTransform(nn.Module):
    """
    Do utterance level preemphasis (we do frame-level preemphasis in STFT layer)
    Args:
        pre_emphasis: preemphasis factor
    """

    def __init__(self, pre_emphasis: float = 0) -> None:
        super(PreEmphasisTransform, self).__init__()
        self.pre_emphasis = pre_emphasis

    def extra_repr(self) -> str:
        return f"pre_emphasis={self.pre_emphasis}"

    def exportable(self) -> bool:
        return False

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


class SpeedPerturbTransform(nn.Module):
    """
    Transform layer for performing speed perturb
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
        if sr not in dst_sr:
            raise ValueError(
                f"We should keep 1.0 in perturb options: {perturb}")
        # N x dst_sr x src_sr x K
        self.weights = nn.ParameterList([
            nn.Parameter(speed_perturb_filter(sr, fs), requires_grad=False)
            for fs in dst_sr
            if fs != sr
        ])
        shapes = [w.shape for w in self.weights]
        self.register_buffer(
            "src_sr", th.tensor([s[1] for s in shapes] + [1], dtype=th.int64))
        self.register_buffer(
            "dst_sr", th.tensor([s[0] for s in shapes] + [1], dtype=th.int64))
        self.last_choice = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sr={self.sr}, factor={self.factor_str})"

    def exportable(self) -> bool:
        return False

    def output_length(self,
                      inp_len: Optional[th.Tensor]) -> Optional[th.Tensor]:
        """
        Compute output length after speed perturb
        """
        if self.last_choice is None:
            return inp_len
        if inp_len is None:
            return None
        return inp_len // self.src_sr[self.last_choice] * self.dst_sr[
            self.last_choice]

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x ... x S
        Return:
            wav (Tensor): output signal, N x ... x S
        """
        self.last_choice = None
        if not self.training:
            return wav
        if wav.dim() != 2:
            raise RuntimeError(f"Now only supports 2D tensor, got {wav.dim()}")
        choice = th.randint(0, len(self.weights) + 1, (wav.shape[0],))
        self.last_choice = choice
        wav_sp = []
        # each utterance is different
        # NOTE: make it same in previous commits
        for i, c in enumerate(self.last_choice.tolist()):
            # 1.0, do not apply speed perturb
            if c == len(self.weights):
                wav_sp.append(wav[i])
            else:
                wav_sp.append(perturb_speed(wav[i:i + 1], self.weights[c])[0])
        # may produce longer utterance
        wav_sp_pad = th.zeros(
            [wav.shape[0], max([w.shape[-1] for w in wav_sp])],
            device=wav.device)
        for i, w in enumerate(wav_sp):
            wav_sp_pad[i, :w.shape[-1]] = w
        return wav_sp_pad


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

    def exportable(self) -> bool:
        return True

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Args:
            tensor (Tensor): input signal, N x ... x F x T
        Return:
            tensor (Tensor): output signal, N x ... x T x F
        """
        return tensor.transpose(-1, -2)


class SpectrogramTransform(STFT):
    """
    (Power|Linear) spectrogram feature extraction
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
                 mode: str = "librosa") -> None:
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

    def dim(self) -> int:
        return self.num_bins

    def exportable(self) -> bool:
        return False

    def forward(self, wav: th.Tensor) -> th.Tensor:
        """
        Args:
            wav (Tensor): input signal, N x (C) x S
        Return:
            mag (Tensor): magnitude, N x (C) x F x T x 2
        """
        # N x (C) x F x T x 2
        return super().forward(wav, return_polar=False)


class MagnitudeTransform(nn.Module):
    """
    Transform tensor [real, imag] to angle tensor
    """

    def __init__(self, dim: int = -1, eps: float = 0):
        super(MagnitudeTransform, self).__init__()
        self.dim = dim
        self.eps = eps

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"

    def exportable(self) -> bool:
        return True

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x ... x 2 x ...
        Return:
            out (Tensor): N x ...
        """
        return th.sqrt(th.sum(inp**2, self.dim) + self.eps)


class AbsTransform(nn.Module):
    """
    Absolute transform
    Args:
        eps: small floor value to avoid NAN when backward
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super(AbsTransform, self).__init__()
        self.eps = eps

    def extra_repr(self) -> str:
        return f"eps={self.eps:.3e}"

    def exportable(self) -> bool:
        return True

    def forward(self, tensor: th.Tensor) -> th.Tensor:
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

    def __init__(self, power: float = 2) -> None:
        super(PowerTransform, self).__init__()
        self.power = power

    def extra_repr(self) -> str:
        return f"power={self.power}"

    def exportable(self) -> bool:
        return True

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Args:
            tensor (Tensor): N x T x F
        Return:
            tensor (Tensor): N x T x F
        """
        return tensor**self.power


class MelTransform(nn.Module):
    """
    Perform mel tranform (multiply mel filters)
    Args:
        frame_len: length of the frame
        round_pow_of_two: if true, choose round(#power_of_two) as the FFT size
        sr: sample rate of souce signal
        num_mels: number of the mel bands
        fmin: lowest frequency (in Hz)
        fmax: highest frequency (in Hz)
        mel_filter: if not "", load mel filter from this
        requires_grad: make it trainable or not
    """

    def __init__(self,
                 frame_len: int,
                 round_pow_of_two: bool = True,
                 sr: int = 16000,
                 num_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None,
                 mel_matrix: str = "",
                 coeff_norm: bool = False,
                 requires_grad: bool = False) -> None:
        super(MelTransform, self).__init__()
        if mel_matrix:
            # pass existed tensor for initialization
            filters = th.load(mel_matrix)
        else:
            # NOTE: the following mel matrix is similiar (not equal to) with
            #       the kaldi results
            filters = mel_filter(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels,
                                 fmax=fmax,
                                 fmin=fmin,
                                 norm=coeff_norm)
        self.num_mels, self.num_bins = filters.shape
        # num_mels x (N // 2 + 1)
        self.filters = nn.Parameter(filters, requires_grad=requires_grad)
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax
        self.init = mel_matrix if mel_matrix else "librosa"

    def dim(self) -> int:
        return self.num_mels

    def exportable(self) -> bool:
        return True

    def extra_repr(self) -> str:
        shape = self.filters.shape
        return (f"fmin={self.fmin}, fmax={self.fmax}, " +
                f"mel_filter={shape[0]}x{shape[1]}, init={self.init}")

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
        fbank = tf.linear(linear, self.filters, bias=None)
        return fbank


class LogTransform(nn.Module):
    """
    Transform feature from linear domain to log domain
    Args:
        eps: floor value to avoid nagative values
        lower_bound: lower bound value
    """

    def __init__(self, eps: float = 1e-5, lower_bound: float = 0.0) -> None:
        super(LogTransform, self).__init__()
        self.eps = eps
        self.lower_bound = lower_bound

    def dim_scale(self) -> int:
        return 1

    def exportable(self) -> bool:
        return True

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
    Perform DCT (for mfcc features)
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
        # num_mels x num_ceps
        dct_mat = scipy_dct(th.eye(num_mels).numpy(),
                            norm="ortho")[:, :num_ceps]
        # num_ceps x num_mels
        # NOTE: DCT matrix is compatiable with kaldi
        self.dct = nn.Parameter(th.from_numpy(dct_mat.T), requires_grad=False)
        if lifter > 0:
            cepstral_lifter = 1 + lifter * 0.5 * th.sin(
                math.pi * th.arange(1, 1 + num_ceps) / lifter)
            self.cepstral_lifter = nn.Parameter(cepstral_lifter,
                                                requires_grad=False)
        else:
            self.cepstral_lifter = None

    def dim(self) -> int:
        return self.num_ceps

    def exportable(self) -> bool:
        return True

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
        mfcc = tf.linear(log_mel, self.dct, bias=None)
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
            gcmvn_toks = gcmvn.split(".")
            # in Kaldi format
            if gcmvn_toks[-1] == "ark":
                cmvn = th.tensor(read_kaldi_mat(gcmvn), dtype=th.float32)
                N = cmvn[0, -1]
                mean = cmvn[0, :-1] / N
                std = (cmvn[1, :-1] / N - mean**2)**0.5
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
            f"norm_mean={self.norm_mean}, norm_var={self.norm_var}, per_band={self.per_band}, "
            + f"gcmvn_stats={self.gcmvn}, eps={self.eps:.3e}")

    def dim_scale(self) -> int:
        return 1

    def exportable(self) -> bool:
        return True

    def _cmvn_per_band(self, feats: th.Tensor) -> th.Tensor:
        if self.norm_mean:
            feats = feats - th.mean(feats, -1, keepdim=True)
        if self.norm_var:
            if self.norm_mean:
                var = th.mean(feats**2, -1, keepdim=True)
            else:
                var = th.var(feats, -1, unbiased=False, keepdim=True)
            feats = feats / th.sqrt(var + self.eps)
        return feats

    def _cmvn_all_band(self, feats: th.Tensor) -> th.Tensor:
        if self.norm_mean:
            feats = feats - th.mean(feats, (-1, -2), keepdim=True)
        if self.norm_var:
            if self.norm_mean:
                var = th.mean(feats**2, (-1, -2), keepdim=True)
            else:
                var = th.var(feats, (-1, -2), unbiased=False, keepdim=True)
            feats = feats / th.sqrt(var + self.eps)
        return feats

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
            # for th.jit.export, we have two similar function
            if self.per_band:
                feats = self._cmvn_per_band(feats)
            else:
                feats = self._cmvn_all_band(feats)
        return feats


class SpecAugTransform(nn.Module):
    """
    Spectra data augmentation
    Args:
        p: probability to do spec-augment
        p_time: p in SpecAugment paper
        time_args: (T, m_T) in the SpecAugment paper
        freq_args: (F, m_F) in the SpecAugment paper
        mask_zero: use zero value or mean in the masked region
    """

    def __init__(self,
                 p: float = 0.5,
                 p_time: float = 1.0,
                 time_args: Tuple[int] = [40, 1],
                 freq_args: Tuple[int] = [30, 1],
                 mask_zero: bool = True) -> None:
        super(SpecAugTransform, self).__init__()
        assert len(freq_args) == 2 and len(time_args) == 2
        self.fnum, self.tnum = freq_args[1], time_args[1]
        self.mask_zero = mask_zero
        self.F, self.T = freq_args[0], time_args[0]
        # prob to do spec-augment
        self.p = p
        # max portion constraint on time axis
        self.p_time = p_time

    def extra_repr(self) -> str:
        return (
            f"max_bands={self.F}, max_frame={self.T}, " +
            f"p={self.p}, p_time={self.p_time}, mask_zero={self.mask_zero}, "
            f"num_freq_masks={self.fnum}, num_time_masks={self.tnum}")

    def exportable(self) -> bool:
        return False

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
    Do feature splicing & subsampling if needed
    Args:
        lctx: left context
        rctx: right context
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

    def exportable(self) -> bool:
        return True

    def forward(self, feats: th.Tensor) -> th.Tensor:
        """
        args:
            feats (Tensor): original feature, N x ... x Ti x F
        return:
            slice (Tensor): spliced feature, N x ... x To x FD
        """
        # N x ... x T x FD
        feats = splice_feature(feats, lctx=self.lctx, rctx=self.rctx)
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

    def __init__(self,
                 ctx: int = 2,
                 order: int = 2,
                 delta_as_channel: bool = False) -> None:
        super(DeltaTransform, self).__init__()
        self.ctx = ctx
        self.order = order
        scale = th.arange(-ctx, ctx + 1, dtype=th.float32)
        normalizer = sum(i * i for i in range(-ctx, ctx + 1))
        self.scale = nn.Parameter(scale / normalizer, requires_grad=False)
        self.delta_as_channel = delta_as_channel

    def extra_repr(self) -> str:
        return f"context={self.ctx}, order={self.order}, delta_as_channel={self.delta_as_channel}"

    def dim_scale(self) -> int:
        return self.order

    def exportable(self) -> bool:
        return True

    def forward(self, feats: th.Tensor) -> th.Tensor:
        """
        args:
            feats (Tensor): original feature, N x (C) x T x F
        return:
            delta (Tensor): delta feature, N x (C) x T x FD
        """
        delta = [feats]
        for _ in range(self.order):
            # N x T x F x (2C+1)
            splice = splice_feature(delta[-1],
                                    lctx=self.ctx,
                                    rctx=self.ctx,
                                    op="stack")
            # N x T x F
            delta.append(th.sum(splice * self.scale, -1))
        if self.delta_as_channel:
            # N x C x T x F
            return th.stack(delta, 1)
        else:
            # N x ... x T x FD
            return th.cat(delta, -1)


@ApsRegisters.transform.register("asr")
class FeatureTransform(nn.Module):
    """
    Feature transform for ASR tasks
        - RescaleTransform
        - PreEmphasisTransform
        - SpeedPerturbTransform
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
        audio_norm: use audio samples normalized between [-1, 1] or [-MAX-INT16, MAX-INT16]
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
        aug_time_args: (T, m_T) in the SpecAugment paper
        aug_freq_args: (F, m_F) in the SpecAugment paper
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
                 audio_norm: bool = True,
                 pre_emphasis: float = 0.97,
                 use_power: bool = False,
                 sr: int = 16000,
                 speed_perturb: str = "0.9,1.0,1.1",
                 log_lower_bound: float = 0,
                 num_mels: int = 80,
                 mel_matrix: str = "",
                 mel_coeff_norm: bool = False,
                 min_freq: int = 0,
                 max_freq: Optional[int] = None,
                 num_ceps: int = 13,
                 lifter: float = 0,
                 aug_prob: float = 0,
                 aug_maxp_time: float = 0.5,
                 aug_mask_zero: bool = True,
                 aug_time_args: Tuple[int] = (40, 1),
                 aug_freq_args: Tuple[int] = (30, 1),
                 norm_mean: bool = True,
                 norm_var: bool = True,
                 norm_per_band: bool = True,
                 gcmvn: str = "",
                 subsampling_factor: int = 1,
                 lctx: int = 1,
                 rctx: int = 1,
                 delta_ctx: int = 2,
                 delta_order: int = 2,
                 delta_as_channel: bool = False,
                 requires_grad: bool = False,
                 eps: float = EPSILON) -> None:
        super(FeatureTransform, self).__init__()
        if not feats:
            raise ValueError("FeatureTransform: \'feats\' can not be empty")
        feat_tokens = feats.split("-")
        transform = [] if audio_norm else [RescaleTransform()]
        feats_dim = 0
        stft_kwargs = {
            "mode": stft_mode,
            "window": window,
            "center": center,
            "normalized": stft_normalized,
            "pre_emphasis": pre_emphasis,
            "round_pow_of_two": round_pow_of_two
        }
        mel_kwargs = {
            "round_pow_of_two": round_pow_of_two,
            "sr": sr,
            "fmin": min_freq,
            "fmax": max_freq,
            "num_mels": num_mels,
            "coeff_norm": mel_coeff_norm,
            "mel_matrix": mel_matrix,
            "requires_grad": requires_grad
        }
        self.spectra_index = -1
        self.perturb_index = -1
        for tok in feat_tokens:
            if tok == "perturb":
                self.perturb_index = len(transform)
                transform.append(
                    SpeedPerturbTransform(sr=sr, perturb=speed_perturb))
            elif tok == "emph":
                transform.append(
                    PreEmphasisTransform(pre_emphasis=pre_emphasis))
            elif tok == "spectrogram":
                self.spectra_index = len(transform)
                spectrogram = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    MagnitudeTransform(dim=-1),
                    TFTransposeTransform(),
                    PowerTransform(power=2 if use_power else 1)
                ]
                transform += spectrogram
                feats_dim = spectrogram[0].dim()
            elif tok == "fbank":
                self.spectra_index = len(transform)
                fbank = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    MagnitudeTransform(dim=-1),
                    TFTransposeTransform(),
                    PowerTransform(power=2 if use_power else 1),
                    MelTransform(frame_len, **mel_kwargs)
                ]
                transform += fbank
                feats_dim = transform[-1].dim()
            elif tok == "mfcc":
                self.spectra_index = len(transform)
                mfcc = [
                    SpectrogramTransform(frame_len, frame_hop, **stft_kwargs),
                    MagnitudeTransform(dim=-1),
                    TFTransposeTransform(),
                    PowerTransform(power=2 if use_power else 1),
                    MelTransform(frame_len, **mel_kwargs),
                    LogTransform(eps=eps),
                    DiscreteCosineTransform(num_ceps=num_ceps,
                                            num_mels=num_mels,
                                            lifter=lifter)
                ]
                transform += mfcc
                feats_dim = transform[-1].dim()
            elif tok == "trans":
                transform.append(TFTransposeTransform())
            elif tok == "pow":
                transform.append(PowerTransform())
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
                                     freq_args=aug_freq_args,
                                     time_args=aug_time_args,
                                     mask_zero=aug_mask_zero))
            elif tok == "splice":
                transform.append(
                    SpliceTransform(lctx=lctx,
                                    rctx=rctx,
                                    subsampling_factor=subsampling_factor))
                feats_dim *= (1 + lctx + rctx)
            elif tok == "delta":
                transform.append(
                    DeltaTransform(ctx=delta_ctx,
                                   order=delta_order,
                                   delta_as_channel=delta_as_channel))
                feats_dim *= (1 + delta_order)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        self.transform = nn.Sequential(*transform)
        self.feats_dim = feats_dim
        self.subsampling_factor = subsampling_factor

    def num_frames(self, inp_len: th.Tensor) -> th.Tensor:
        """
        Work out number of frames
        """
        if inp_len is None:
            return None
        if self.spectra_index == -1:
            warnings.warn("SpectrogramTransform layer is not found, " +
                          "return input as the #num_frames")
            return inp_len
        if self.perturb_index != -1:
            inp_len = self.transform[self.perturb_index].output_length(inp_len)
        num_frames = self.transform[self.spectra_index].num_frames(inp_len)
        return num_frames // self.subsampling_factor

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> AsrReturnType:
        """
        Args:
            inp_pad (Tensor): raw waveform or feature: N x C x S or N x S
            inp_len (Tensor or None): N or None
        Return:
            feats (Tensor): acoustic features: N x C x T x ...
            num_frames (Tensor or None): number of frames
        """
        feats = self.transform(inp_pad)
        num_frames = self.num_frames(inp_len)
        return check_valid(feats, num_frames)
