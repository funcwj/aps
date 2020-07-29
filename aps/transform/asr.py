#!/usr/bin/env python

# wujian@2019
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

from aps.transform.utils import STFT, EPSILON, init_melfilter, init_dct
from aps.transform.spec_aug import tf_mask


class SpectrogramTransform(STFT):
    """
    Compute spectrogram as a layer
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="hamm",
                 round_pow_of_two=True,
                 normalized=False,
                 onesided=True):
        super(SpectrogramTransform,
              self).__init__(frame_len,
                             frame_hop,
                             window=window,
                             round_pow_of_two=round_pow_of_two,
                             normalized=normalized,
                             onesided=onesided)

    def dim(self):
        return self.num_bins

    def len(self, xlen):
        return self.num_frames(xlen)

    def forward(self, x):
        """
        Args:
            x (Tensor): input signal, N x (C) x S
        Return:
            m (Tensor): magnitude, N x (C) x T x F
        """
        m, _ = super().forward(x)
        m = th.transpose(m, -1, -2)
        return m


class AbsTransform(nn.Module):
    """
    Absolute transform
    """
    def __init__(self, eps=1e-5):
        super(AbsTransform, self).__init__()
        self.eps = eps

    def extra_repr(self):
        return f"eps={self.eps:.3e}"

    def forward(self, x):
        """
        Args:
            x (Tensor or ComplexTensor): N x T x F
        Return:
            y (Tensor): N x T x F
        """
        if not isinstance(x, th.Tensor):
            x = x + self.eps
        return x.abs()


class MelTransform(nn.Module):
    """
    Mel tranform as a layer
    """
    def __init__(self,
                 frame_len,
                 round_pow_of_two=True,
                 sr=16000,
                 num_mels=80,
                 fmin=0.0,
                 fmax=None,
                 requires_grad=False):
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

    def dim(self):
        return self.num_mels

    def extra_repr(self):
        return "fmin={0}, fmax={1}, mel_filter={2[0]}x{2[1]}".format(
            self.fmin, self.fmax, self.filters.shape)

    def forward(self, x):
        """
        Args:
            x (Tensor): spectrogram, N x (C) x T x F
        Return:
            f (Tensor): mel-fbank feature, N x (C) x T x B
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError("MelTransform expect 3/4D tensor, " +
                               f"but got {x.dim():d} instead")
        # N x T x F => N x T x M
        f = F.linear(x, self.filters, bias=None)
        return f


class LogTransform(nn.Module):
    """
    Transform linear domain to log domain
    """
    def __init__(self, eps=1e-5):
        super(LogTransform, self).__init__()
        self.eps = eps

    def dim_scale(self):
        return 1

    def extra_repr(self):
        return f"eps={self.eps:.3e}"

    def forward(self, x):
        """
        Args:
            x (Tensor): linear, N x (C) x T x F
        Return:
            y (Tensor): log features, N x (C) x T x F
        """
        x = th.clamp(x, min=self.eps)
        return th.log(x)


class DiscreteCosineTransform(nn.Module):
    """
    DCT as a layer (for mfcc features)
    """
    def __init__(self, num_ceps=13, num_mels=40, lifter=0):
        super(DiscreteCosineTransform, self).__init__()
        self.lifter = lifter
        self.num_ceps = num_ceps
        self.dct = nn.Parameter(init_dct(num_ceps, num_mels),
                                requires_grad=False)
        if lifter > 0:
            cepstral_lifter = 1 + lifter * 0.5 * th.sin(
                math.pi * th.arange(1, 1 + num_ceps) / lifter)
            self.cepstral_lifter = nn.Parameter(cepstral_lifter,
                                                requires_grad=False)
        else:
            self.cepstral_lifter = None

    def dim(self):
        return self.num_ceps

    def extra_repr(self):
        return "cepstral_lifter={0}, dct={1[0]}x{1[1]}".format(
            self.lifter, self.dct.shape)

    def forward(self, x):
        """
        Args:
            x (Tensor): log mel-fbank, N x (C) x T x B
        Return:
            f (Tensor): mfcc, N x (C) x T x P
        """
        f = F.linear(x, self.dct, bias=None)
        if self.cepstral_lifter is not None:
            f = f * self.cepstral_lifter
        return f


class CmvnTransform(nn.Module):
    """
    Utterance-level mean-variance normalization
    """
    def __init__(self, norm_mean=True, norm_var=True, gcmvn="", eps=1e-5):
        super(CmvnTransform, self).__init__()
        self.gmean, self.gstd = None, None
        if gcmvn:
            stats = np.load(gcmvn)
            mean, std = stats[0], stats[1]
            self.gmean = nn.Parameter(mean, requires_grad=False)
            self.gstd = nn.Parameter(std, requires_grad=False)
        self.norm_mean = norm_mean
        self.norm_var = norm_var
        self.gcmvn = gcmvn
        self.eps = eps

    def extra_repr(self):
        return f"norm_mean={self.norm_mean}, norm_var={self.norm_var}, " + \
            f"gcmvn_stats={self.gcmvn}, eps={self.eps:.3e}"

    def dim_scale(self):
        return 1

    def forward(self, x):
        """
        Args:
            x (Tensor): feature before normalization, N x (C) x T x F
        Return:
            y (Tensor): normalized feature, N x (C) x T x F
        """
        if not self.norm_mean and not self.norm_var:
            return x
        # over time axis
        m = th.mean(x, -2, keepdim=True) if self.gmean is None else self.gmean
        if self.gstd is None:
            ms = th.mean(x**2, -2, keepdim=True)
            s = (ms - m**2 + self.eps)**0.5
        else:
            s = self.gstd
        if self.norm_mean:
            x = x - m
        if self.norm_var:
            x = x / s
        return x


class SpecAugTransform(nn.Module):
    """
    Spectra data augmentation
    """
    def __init__(self,
                 p=0.5,
                 max_bands=30,
                 max_frame=40,
                 num_freq_masks=2,
                 num_time_masks=2):
        super(SpecAugTransform, self).__init__()
        self.fnum, self.tnum = num_freq_masks, num_time_masks
        self.F, self.T = max_bands, max_frame
        self.p = p

    def extra_repr(self):
        return f"max_bands={self.F}, max_frame={self.T}, p={self.p}, " \
                + f"num_freq_masks={self.fnum}, num_time_masks={self.tnum}"

    def forward(self, x):
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
                           max_bands=self.F,
                           max_frame=self.T,
                           num_freq_masks=self.fnum,
                           num_time_masks=self.tnum,
                           device=x.device)
            if x.dim() == 4:
                # N x 1 x T x F
                mask = mask.unsqueeze(1)
            x = x * mask
        return x


class SpliceTransform(nn.Module):
    """
    Do splicing as well as downsampling if needed
    """
    def __init__(self, lctx=0, rctx=0, ds_rate=1):
        super(SpliceTransform, self).__init__()
        self.rate = ds_rate
        self.lctx = max(lctx, 0)
        self.rctx = max(rctx, 0)

    def extra_repr(self):
        return f"context=({self.lctx}, {self.rctx}), downsample_rate={self.rate}"

    def dim_scale(self):
        return (1 + self.rctx + self.lctx)

    def forward(self, x):
        """
        args:
            x (Tensor): original feature, N x ... x Ti x F
        return:
            y (Tensor): spliced feature, N x ... x To x FD
        """
        T = x.shape[-2]
        T = T - T % self.rate
        if self.lctx + self.rctx != 0:
            ctx = []
            for c in range(-self.lctx, self.rctx + 1):
                idx = th.arange(c, c + T, device=x.device, dtype=th.int64)
                idx = th.clamp(idx, min=0, max=T - 1)
                # N x ... x T x F
                ctx.append(th.index_select(x, -2, idx))
            # N x ... x T x FD
            x = th.cat(ctx, -1)
        if self.rate != 1:
            x = x[..., ::self.rate, :]
        return x


class DeltaTransform(nn.Module):
    """
    Add delta features
    """
    def __init__(self, ctx=2, order=2):
        super(DeltaTransform, self).__init__()
        self.ctx = ctx
        self.order = order

    def extra_repr(self):
        return f"context={self.ctx}, order={self.order}"

    def dim_scale(self):
        return self.order

    def _add_delta(self, x):
        dx = th.zeros_like(x)
        for i in range(1, self.ctx + 1):
            dx[..., :-i, :] += i * x[..., i:, :]
            dx[..., i:, :] += -i * x[..., :-i, :]
            dx[..., -i:, :] += i * x[..., -1:, :]
            dx[..., :i, :] += -i * x[..., :1, :]
        dx = dx / (2 * sum(i**2 for i in range(1, self.ctx + 1)))
        return dx

    def forward(self, x):
        """
        args:
            x (Tensor): original feature, N x (C) x T x F
        return:
            y (Tensor): delta feature, N x (C) x T x FD
        """
        delta = [x]
        for _ in range(self.order):
            delta.append(self._add_delta(delta[-1]))
        # N x ... x T x FD
        return th.cat(delta, -1)


class FeatureTransform(nn.Module):
    """
    Feature transform for ASR tasks
        - Spectrogram 
        - MelTransform
        - AbsTransform
        - LogTransform 
        - DiscreteCosineTransform
        - CmvnTransform 
        - SpecAugTransform 
        - SpliceTransform
        - DeltaTransform
    """
    def __init__(self,
                 feats="fbank-log-cmvn",
                 frame_len=400,
                 frame_hop=160,
                 window="hamm",
                 round_pow_of_two=True,
                 stft_normalized=False,
                 sr=16000,
                 num_mels=80,
                 num_ceps=13,
                 lifter=0,
                 aug_prob=0,
                 aug_max_bands=30,
                 aug_max_frame=40,
                 num_aug_bands=2,
                 num_aug_frame=2,
                 norm_mean=True,
                 norm_var=True,
                 gcmvn="",
                 ds_rate=1,
                 lctx=1,
                 rctx=1,
                 delta_ctx=2,
                 delta_order=2,
                 requires_grad=False,
                 eps=EPSILON):
        super(FeatureTransform, self).__init__()
        trans_tokens = feats.split("-")
        transform = []
        feats_dim = 0
        downsample_rate = 1
        for tok in trans_tokens:
            if tok == "spectrogram":
                transform.append(
                    SpectrogramTransform(frame_len,
                                         frame_hop,
                                         window=window,
                                         round_pow_of_two=round_pow_of_two,
                                         normalized=stft_normalized))
                feats_dim = transform[-1].dim()
            elif tok == "fbank":
                fbank = [
                    SpectrogramTransform(frame_len,
                                         frame_hop,
                                         window=window,
                                         round_pow_of_two=round_pow_of_two,
                                         normalized=stft_normalized),
                    MelTransform(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels,
                                 requires_grad=requires_grad)
                ]
                transform += fbank
                feats_dim = transform[-1].dim()
            elif tok == "mfcc":
                log_fbank = [
                    SpectrogramTransform(frame_len,
                                         frame_hop,
                                         window=window,
                                         round_pow_of_two=round_pow_of_two),
                    MelTransform(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels),
                    LogTransform(eps=eps),
                    DiscreteCosineTransform(num_ceps=num_ceps,
                                            num_mels=num_mels,
                                            lifter=lifter)
                ]
                transform += log_fbank
            elif tok == "mel":
                transform.append(
                    MelTransform(frame_len,
                                 round_pow_of_two=round_pow_of_two,
                                 sr=sr,
                                 num_mels=num_mels))
                feats_dim = transform[-1].dim()
            elif tok == "log":
                transform.append(LogTransform(eps=eps))
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
                                  gcmvn=gcmvn,
                                  eps=eps))
            elif tok == "aug":
                transform.append(
                    SpecAugTransform(p=aug_prob,
                                     max_bands=aug_max_bands,
                                     max_frame=aug_max_frame,
                                     num_freq_masks=num_aug_bands,
                                     num_time_masks=num_aug_frame))
            elif tok == "splice":
                transform.append(
                    SpliceTransform(lctx=lctx, rctx=rctx, ds_rate=ds_rate))
                feats_dim *= (1 + lctx + rctx)
                downsample_rate = ds_rate
            elif tok == "delta":
                transform.append(
                    DeltaTransform(ctx=delta_ctx, order=delta_order))
                feats_dim *= (1 + delta_order)
            else:
                raise RuntimeError(f"Unknown token {tok} in {feats}")
        self.transform = nn.Sequential(*transform)
        self.feats_dim = feats_dim
        self.downsample_rate = downsample_rate

    def num_frames(self, x_len):
        """
        Work out number of frames
        """
        if not isinstance(self.transform[0], SpectrogramTransform):
            raise RuntimeError(
                "0-th layer of transform is not SpectrogramTransform")
        return self.transform[0].len(x_len)

    def forward(self, x_pad, x_len):
        """
        Args:
            x_pad (Tensor): raw waveform: N x C x S or N x S
            x_len (Tensor or None): N or None
        Return:
            feats_pad (Tensor): acoustic features: N x C x T x ...
            feats_len (Tensor or None): number of frames
        """
        feats_pad = self.transform(x_pad)
        if x_len is None:
            feats_len = None
        else:
            feats_len = self.num_frames(x_len)
            feats_len = feats_len // self.downsample_rate
        return feats_pad, feats_len