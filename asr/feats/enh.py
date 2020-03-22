#!/usr/bin/env python

# wujian@2019
"""
Feature transform for Enhancement/Separation
"""
import math

import torch as th
import torch.nn as nn

from torch_complex.tensor import ComplexTensor

from .utils import STFT, EPSILON
from .asr import LogTransform, CmvnTransform, SpecAugTransform

MATH_PI = math.pi


class IpdTransform(nn.Module):
    """
    Compute inter-channel phase difference
    """
    def __init__(self, ipd_index="1,0", cos=True, sin=False):
        super(IpdTransform, self).__init__()
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f"ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}"

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
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


class FeatureTransform(nn.Module):
    """
    Feature transform for ASR tasks
    Spectrogram - LogTransform - CmvnTransform + IpdTransform
    """
    def __init__(self,
                 feats="spectrogram-log-cmvn-ipd",
                 frame_len=512,
                 frame_hop=256,
                 window="sqrthann",
                 round_pow_of_two=True,
                 sr=16000,
                 gcmvn="",
                 norm_mean=True,
                 norm_var=True,
                 aug_prob=0,
                 aug_max_bands=90,
                 aug_max_frame=40,
                 num_aug_bands=2,
                 num_aug_frame=2,
                 ipd_index="1,0",
                 cos_ipd=True,
                 sin_ipd=False,
                 eps=EPSILON):
        super(FeatureTransform, self).__init__()
        self.STFT = STFT(frame_len,
                         frame_hop,
                         window=window,
                         round_pow_of_two=round_pow_of_two)
        trans_tokens = feats.split("-") if feats else []
        transform = []
        feats_dim = 0
        feats_ipd = None
        for i, tok in enumerate(trans_tokens):
            if i == 0:
                if tok != "spectrogram" and tok != "ipd":
                    raise RuntimeError("Now only support spectrogram features")
                feats_dim = self.STFT.num_bins
            elif tok == "log":
                transform.append(LogTransform(eps=EPSILON))
            elif tok == "cmvn":
                transform.append(
                    CmvnTransform(norm_mean=norm_mean,
                                  norm_var=norm_var,
                                  gcmvn=gcmvn,
                                  eps=eps))
            elif tok == "ipd":
                feats_ipd = IpdTransform(ipd_index=ipd_index,
                                         cos=cos_ipd,
                                         sin=sin_ipd)
                base = 0 if i == 0 else 1
                if cos_ipd and sin_ipd:
                    feats_dim *= len(ipd_index) * (2 + base)
                else:
                    feats_dim *= len(ipd_index) * (1 + base)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        if len(transform):
            self.spe_transform = nn.Sequential(*transform)
        else:
            self.spe_transform = None
        self.ipd_transform = feats_ipd
        if aug_prob > 0:
            self.aug_transform = SpecAugTransform(p=aug_prob,
                                                  max_bands=aug_max_bands,
                                                  max_frame=aug_max_frame,
                                                  num_freq_masks=num_aug_bands,
                                                  num_time_masks=num_aug_frame)
        else:
            self.aug_transform = None
        self.feats_dim = feats_dim

    def forward(self, x_pad, x_len, norm_obs=False):
        """
        args:
            x_pad: raw waveform: N x C x S or N x S
            x_len: N or None
        return:
            feats: spatial+spectral features: N x T x ...
            f_len: N or None
        """
        # N x C x F x T
        mag, pha = self.STFT(x_pad)
        # spectra transform
        if self.spe_transform:
            # N x T x F
            feats = mag[:, 0].transpose(-1, -2)
            # spectra features of CH0, N x T x F
            feats = self.spe_transform(feats)
            if self.aug_transform:
                # spectra augmentation if needed
                feats = self.aug_transform(feats)
        else:
            feats = None
            # spectra augmentation if needed
            if self.aug_transform:
                mag = self.aug_transform(mag)
        # complex spectrogram of CH 0~(C-1), N x C x F x T
        if norm_obs:
            mag_norm = th.norm(mag, p=2, dim=1, keepdim=True)
            mag = mag / th.clamp(mag_norm, min=EPSILON)
        real = mag * th.cos(pha)
        imag = mag * th.sin(pha)
        cplx = ComplexTensor(real, imag)
        # ipd transform
        if self.ipd_transform:
            # N x T x ...
            ipd = self.ipd_transform(pha)
            # N x ... x T
            ipd = ipd.transpose(1, 2)
            # N x T x ...
            if feats is not None:
                feats = th.cat([feats, ipd], -1)
            else:
                feats = ipd
        f_len = self.STFT.num_frames(x_len) if x_len is not None else None
        return feats, cplx, f_len