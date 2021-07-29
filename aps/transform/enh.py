#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Spatial feature transform (for Speech Separation & Enhancement), using enh.py for abbreviation
"""
import math

import torch as th
import torch.nn as nn

from typing import Union, List, Optional, Tuple
from aps.transform.utils import STFT, iSTFT
from aps.transform.asr import (TFTransposeTransform, LogTransform,
                               CmvnTransform, SpecAugTransform)
from aps.transform.asr import check_valid
from aps.const import MATH_PI, EPSILON
from aps.libs import ApsRegisters
from aps.cplx import ComplexTensor

EnhReturnType = Tuple[th.Tensor, ComplexTensor, Optional[th.Tensor]]


class RefChannelTransform(nn.Module):
    """
    Choose one reference channel
    """

    def __init__(self, ref_channel: int = 0, input_dim: int = 4) -> None:
        super(RefChannelTransform, self).__init__()
        # < 0 means return all
        self.ref_channel = ref_channel
        self.input_dim = input_dim

    def extra_repr(self) -> str:
        return f"ref_channel={self.ref_channel}"

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x (C) x ...
        Return:
            out (Tensor): N x ...
        """
        if inp.dim() != self.input_dim or self.ref_channel < 0:
            return inp
        else:
            return inp[:, self.ref_channel]


class AngleTransform(nn.Module):
    """
    Transform tensor [real, imag] to angle tensor
    """

    def __init__(self, dim: int = -1):
        super(AngleTransform, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x ... x 2 x ...
        Return:
            out (Tensor): N x ...
        """
        real = th.select(inp, self.dim, 0)
        imag = th.select(inp, self.dim, 1)
        return th.atan2(imag, real)


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

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x ... x 2 x ...
        Return:
            out (Tensor): N x ...
        """
        return th.sqrt(th.sum(inp**2, self.dim) + self.eps)


class IpdTransform(nn.Module):
    """
    Compute inter-channel phase difference (IPD) features
    Args:
        ipd_index: index pairs to compute IPD feature
        cos: using cosIPD or not
        sin: adding sinIPD or not
    """

    def __init__(self,
                 ipd_index: str = "1,0",
                 cos: bool = True,
                 sin: bool = False) -> None:
        super(IpdTransform, self).__init__()

        def split_index(sstr):
            return [tuple(map(int, p.split(","))) for p in sstr.split(";")]

        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self) -> str:
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p: th.Tensor) -> th.Tensor:
        """
        Accept multi-channel phase and output inter-channel phase difference
        Args
            p (Tensor): phase matrix, N x C x F x T
        Return
            ipd (Tensor): IPD features,  N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__class__.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.cos:
            # N x M x F x T
            ipd = th.cos(pha_dif)
            if self.sin:
                # N x M x 2F x T, along frequency axis
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            # ipd = th.fmod(pha_dif + math.pi, 2 * math.pi) - math.pi
            ipd = th.where(ipd > MATH_PI, ipd - MATH_PI * 2, ipd)
            ipd = th.where(ipd <= -MATH_PI, ipd + MATH_PI * 2, ipd)
        # N x MF x T
        ipd = ipd.view(N, -1, T)
        # N x MF x T
        return ipd


class DfTransform(nn.Module):
    """
    Compute angle/directional feature
        1) num_doas == 1: we known the DoA of the target speaker
        2) num_doas != 1: we do not have that prior, so we sampled #num_doas DoAs
                          and compute on each directions
    Note that the angle feature is geometry dependent. To to support your own microphone arrays, the user
    should re-implement the function _oracle_phase_delay(...), which returns the oracle phase difference
    based on the given microphone topology

    Args:
        geometric: name of the array geometry
        sr: sample rate of the audio
        velocity: sound speed
        num_bins: number of FFT bins
        num_doas: how many directions to point out
        af_index: index pairs to compute the directional feature
    """

    def __init__(self,
                 geometric: str = "7@",
                 sr: int = 16000,
                 velocity: int = 340,
                 num_bins: int = 257,
                 num_doas: int = 1,
                 af_index: str = "1,0;2,0;3,0;4,0;5,0;6,0") -> None:
        super(DfTransform, self).__init__()
        # NOTE: add your costomized topo here
        if geometric not in ["7@"]:
            raise RuntimeError(f"Unsupported array geometric: {geometric}")
        self.geometric = geometric
        self.sr = sr
        self.num_bins = num_bins
        self.num_doas = num_doas
        self.velocity = velocity

        def split_index(sstr):
            return [tuple(map(int, p.split(","))) for p in sstr.split(";")]

        # ipd index
        pair = split_index(af_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.af_index = af_index
        omega = th.tensor(
            [math.pi * sr * f / (num_bins - 1) for f in range(num_bins)])
        # 1 x F
        self.omega = nn.Parameter(omega[None, :], requires_grad=False)

    def _oracle_phase_delay(self, doa: th.Tensor) -> th.Tensor:
        """
        Compute oracle phase delay given DoA
        Args
            doa (Tensor): N
        Return
            phi (Tensor): N x (D) x C x F
        """
        device = doa.device
        if self.num_doas != 1:
            # doa is a unused, fake parameter
            N = doa.shape[0]
            # N x D
            doa = th.linspace(0, MATH_PI * 2, self.num_doas + 1,
                              device=device)[:-1].repeat(N, 1)
        #      *3    *2
        #
        #   *4    *0    *1
        #
        #      *5    *6
        if self.geometric == "7@":
            R = 0.0425
            zero = th.zeros_like(doa)
            # N x 7 or N x D x 7
            tau = R * th.stack([
                zero, -th.cos(doa), -th.cos(MATH_PI / 3 - doa),
                -th.cos(2 * MATH_PI / 3 - doa),
                th.cos(doa),
                th.cos(MATH_PI / 3 - doa),
                th.cos(2 * MATH_PI / 3 - doa)
            ],
                               dim=-1) / self.velocity
            # (Nx7x1) x (1xF) => Nx7xF or (NxDx7x1) x (1xF) => NxDx7xF
            phi = th.matmul(tau.unsqueeze(-1), -self.omega)
            return phi
        else:
            return None

    def extra_repr(self) -> str:
        return (
            f"geometric={self.geometric}, af_index={self.af_index}, " +
            f"sr={self.sr}, num_bins={self.num_bins}, velocity={self.velocity}, "
            + f"known_doa={self.num_doas == 1}")

    def _compute_af(self, ipd: th.Tensor, doa: th.Tensor) -> th.Tensor:
        """
        Compute angle feature
        Args
            ipd (Tensor): N x C x F x T
            doa (Tensor): DoA of the target speaker (if we known that), N
                 or N x D (we do not known that, sampling D DoAs instead)
        Return
            af (Tensor): N x (D) x F x T
        """
        # N x C x F or N x D x C x F
        d = self._oracle_phase_delay(doa)
        d = d.unsqueeze(-1)
        if self.num_doas == 1:
            dif = d[:, self.index_l] - d[:, self.index_r]
            # N x C x F x T
            af = th.cos(ipd - dif)
            # on channel dimention (mean or sum)
            af = th.mean(af, dim=1)
        else:
            # N x D x C x F x 1
            dif = d[:, :, self.index_l] - d[:, :, self.index_r]
            # N x D x C x F x T
            af = th.cos(ipd.unsqueeze(1) - dif)
            # N x D x F x T
            af = th.mean(af, dim=2)
        return af

    def forward(self, p: th.Tensor, doa: Union[th.Tensor,
                                               List[th.Tensor]]) -> th.Tensor:
        """
        Accept doa of the speaker & multi-channel phase, output angle feature
        Args
            doa (Tensor or list[Tensor]): DoA of target/each speaker, N or [N, ...]
            p (Tensor): phase matrix, N x C x F x T
        Return
            af (Tensor): angle feature, N x F* x T or N x D x F x T (known_doa=False)
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__class__.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        ipd = p[:, self.index_l] - p[:, self.index_r]

        if isinstance(doa, list):
            if self.num_doas != 1:
                raise RuntimeError("known_doa=False, no need to pass "
                                   "doa as a Sequence object")
            # [N x F x T or N x D x F x T, ...]
            af = [self._compute_af(ipd, spk_doa) for spk_doa in doa]
            # N x F x T => N x F* x T
            af = th.cat(af, 1)
        else:
            # N x F x T or N x D x F x T
            af = self._compute_af(ipd, doa)
        return af


class FixedBeamformer(nn.Module):
    """
    Fixed beamformer as a layer
    Args:
        num_beams: number of beams
        num_channels: number of the channels
        num_bins: FFT size / 2 + 1
        weight: beamformer's coefficient
        requires_grad: make it trainable or not
    """

    def __init__(self,
                 num_beams: int,
                 num_channels: int,
                 num_bins: int,
                 weight: Optional[str] = None,
                 requires_grad: bool = False) -> None:
        super(FixedBeamformer, self).__init__()
        if weight:
            # (2, num_directions, num_channels, num_bins)
            w = th.load(weight)
            if w.shape[1] != num_beams:
                raise RuntimeError(f"Number of beam got from {w.shape[1]} " +
                                   f"don't match parameter {num_beams}")
            self.init_weight = weight
        else:
            self.init_weight = None
            w = th.zeros(2, num_beams, num_channels, num_bins)
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        # (num_directions, num_channels, num_bins, 1)
        self.real = nn.Parameter(w[0].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.imag = nn.Parameter(w[1].unsqueeze(-1),
                                 requires_grad=requires_grad)
        self.requires_grad = requires_grad

    def extra_repr(self) -> str:
        B, M, F, _ = self.real.shape
        return (f"num_beams={B}, num_channels={M}, " +
                f"num_bins={F}, init_weight={self.init_weight}, " +
                f"requires_grad={self.requires_grad}")

    def forward(
            self,
            x: ComplexTensor,
            beam: Optional[th.Tensor] = None,
            squeeze: bool = False,
            trans: bool = False,
            cplx: bool = True
    ) -> Union[ComplexTensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Args:
            x (Complex Tensor): N x C x F x T
            beam (Tensor or None): N
        Return:
            1) (Tensor, Tensor): N x (B) x F x T
            2) (ComplexTensor): N x (B) x F x T
        """
        r, i = x.real, x.imag
        if r.dim() != i.dim() and r.dim() != 4:
            raise RuntimeError(f"FixBeamformer accept 4D tensor, got {r.dim()}")
        if self.real.shape[1] != r.shape[1]:
            raise RuntimeError(f"Number of channels mismatch: "
                               f"{r.shape[1]} vs {self.real.shape[1]}")
        if beam is None:
            # output all the beam
            br = th.sum(r.unsqueeze(1) * self.real, 2) + th.sum(
                i.unsqueeze(1) * self.imag, 2)
            bi = th.sum(i.unsqueeze(1) * self.real, 2) - th.sum(
                r.unsqueeze(1) * self.imag, 2)
        else:
            # output selected beam
            br = th.sum(r * self.real[beam], 1) + th.sum(i * self.imag[beam], 1)
            bi = th.sum(i * self.real[beam], 1) - th.sum(r * self.imag[beam], 1)
        if squeeze:
            br = br.squeeze()
            bi = bi.squeeze()
        if trans:
            br = br.transpose(-1, -2)
            bi = bi.transpose(-1, -2)
        if cplx:
            return ComplexTensor(br, bi)
        else:
            return br, bi


@ApsRegisters.transform.register("enh")
class FeatureTransform(nn.Module):
    """
    Feature transform for Enhancement/Separation tasks
    Spectrogram - LogTransform - CmvnTransform + IpdTransform + DfTransform

    Args:
        feats: string that shows the way to extract features
        frame_len: length of the frame
        frame_hop: hop size between frames
        window: window name
        center: center flag (similar with that in librosa.stft)
        round_pow_of_two: if true, choose round(#pow_of_two) as the FFT size
        stft_normalized: use normalized DFT kernel
        stft_mode: "kaldi"|"librosa", slight difference on windowing
        ref_channel: choose one channel
        sr: sample rate of the audio
        norm_mean|norm_var: normalize mean/var or not (cmvn)
        norm_per_band: do cmvn per-band or not (cmvn)
        gcmvn: global cmvn statistics (cmvn)
        aug_prob: probability to do spec-augment
        aug_maxp_time: p in SpecAugment paper
        aug_mask_zero: use zero value or mean in the masked region
        aug_time_args: (T, m_T) in the SpecAugment paper
        aug_freq_args: (F, m_F) in the SpecAugment paper
        ipd_index: index pairs to compute IPD feature (ipd)
        cos_ipd|sin_ipd: using cos or sin IPDs
        eps: floor number
    """

    def __init__(self,
                 feats: str = "spectrogram-log-cmvn",
                 frame_len: int = 512,
                 frame_hop: int = 256,
                 window: str = "sqrthann",
                 round_pow_of_two: bool = True,
                 stft_normalized: bool = False,
                 stft_mode: str = "librosa",
                 center: bool = False,
                 sr: int = 16000,
                 ref_channel: int = 0,
                 gcmvn: str = "",
                 norm_mean: bool = True,
                 norm_var: bool = True,
                 norm_per_band: bool = True,
                 aug_prob: float = 0,
                 aug_maxp_time: float = 1.0,
                 aug_mask_zero: bool = True,
                 aug_time_args: Tuple[int] = (40, 1),
                 aug_freq_args: Tuple[int] = (30, 1),
                 ipd_index: str = "",
                 cos_ipd: bool = True,
                 sin_ipd: bool = False,
                 eps: float = EPSILON) -> None:
        super(FeatureTransform, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.stft_kwargs = {
            "mode": stft_mode,
            "window": window,
            "center": center,
            "normalized": stft_normalized,
            "round_pow_of_two": round_pow_of_two
        }
        self.forward_stft = self.ctx(name="forward_stft")
        self.inverse_stft = self.ctx(name="inverse_stft")

        trans_tokens = feats.split("-") if feats else []
        transform = []
        feats_dim = 0
        self.ipd_transform = None
        for i, tok in enumerate(trans_tokens):
            if i == 0:
                if tok != "spectrogram" and tok != "ipd":
                    raise RuntimeError("Now only support spectrogram features "
                                       "or IPD features")
                feats_dim = self.forward_stft.num_bins
            if tok == "spectrogram":
                spectrogram = [
                    MagnitudeTransform(),
                    RefChannelTransform(ref_channel=ref_channel, input_dim=4),
                    TFTransposeTransform()
                ]
                transform += spectrogram
            elif tok == "log":
                transform.append(LogTransform(eps=EPSILON))
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
            elif tok == "ipd":
                self.ipd_transform = nn.Sequential(
                    AngleTransform(),
                    IpdTransform(ipd_index=ipd_index, cos=cos_ipd, sin=sin_ipd),
                    TFTransposeTransform())
                ipd_index = ipd_index.split(";")
                base = 0 if i == 0 else 1
                if cos_ipd and sin_ipd:
                    feats_dim *= (len(ipd_index) * 2 + base)
                else:
                    feats_dim *= (len(ipd_index) + base)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        if len(transform):
            self.mag_transform = nn.Sequential(*transform)
        else:
            self.mag_transform = None
        self.feats_dim = feats_dim

    def ctx(self, name: str = "forward_stft") -> nn.Module:
        """
        Return ctx(STFT/iSTFT) for task defined in src/aps/task
        """
        ctx = {"forward_stft": STFT, "inverse_stft": iSTFT}
        if name not in ctx:
            raise ValueError(f"Unknown task context: {name}")
        return ctx[name](self.frame_len, self.frame_hop, **self.stft_kwargs)

    def num_frames(self, wav_len: th.Tensor) -> th.Tensor:
        """
        Work out number of frames
        """
        if wav_len is None:
            return None
        # pass to forward_stft class
        return self.forward_stft.num_frames(wav_len)

    def forward(self, wav_pad: th.Tensor,
                wav_len: Optional[th.Tensor]) -> EnhReturnType:
        """
        Args:
            wav_pad (Tensor): raw waveform, N x C x S or N x S
            wav_len (Tensor or None): number samples in wav_pad, N or None
        Return:
            feats (Tensor): spatial + spectral features, N x T x ...
            cplx (ComplexTensor): STFT coefficients, N x (C) x F x T
            num_frames (Tensor or None): number frames in each batch, N or None
        """
        # packed: N x C x F x T x 2
        packed = self.forward_stft(wav_pad, return_polar=False)
        real, imag = packed[..., 0], packed[..., 1]
        # STFT coefficients: N x C x F x T
        cplx = ComplexTensor(real, imag, polar=False)

        feats = []
        # magnitude transform
        if self.mag_transform:
            # N x (C) x T x F => N x T x F
            feats.append(self.mag_transform(packed))
        # ipd transform
        if self.ipd_transform:
            # N x C x F x T => N x ... x T
            feats.append(self.ipd_transform(packed))
        # concatenate: N x T x ...
        num_frames = self.num_frames(wav_len)
        if len(feats):
            feats = check_valid(th.cat(feats, -1), num_frames)[0]
        else:
            feats = None
        return feats, cplx, num_frames
