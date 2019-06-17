#!/usr/bin/env python

# wujian@2019

import math

import torch as th

import torch.nn.functional as F
import torch.nn as nn

import librosa.filters as libf

EPSILON = th.finfo(th.float32).eps


def init_window(wnd, frame_len):
    if wnd == "sqrt_hann":
        W = th.hann_window(frame_len)**0.5
    elif wnd == "hann":
        W = th.hann_window(frame_len)
    elif wnd == "hamm":
        W = th.hamming_window(frame_len)
    elif wnd == "blackman":
        W = th.blackman_window(frame_len)
    else:
        raise RuntimeError("Unknown window type: {}".format(wnd))
    return W


def init_kernel(frame_len,
                frame_hop,
                round_pow_of_two=True,
                window="sqrt_hann"):
    # window
    W = init_window(window, frame_len)
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N), 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


def init_melmat(frame_len,
                round_pow_of_two=True,
                sr=16000,
                num_mels=80,
                fmin=0.0,
                fmax=None):
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # fmin & fmax
    fmax = sr // 2 if fmax is None else min(fmax, sr // 2)
    # mel-matrix
    mel = libf.mel(sr, N, n_mels=num_mels, fmax=fmax, fmin=fmin, htk=True)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


class Spectrogram(nn.Module):
    """
    Compute spectrogram as a layer
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 round_pow_of_two=True):
        super(Spectrogram, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.stride = frame_hop
        self.window = window
        self.num_bins = K.shape[-1]

    def extra_repr(self):
        return "window={0}, stride={1}, kernel_size={2[0]}x{2[2]}".format(
            self.window, self.stride, self.K.shape)

    def forward(self, x):
        """
        x: input signal, N x 1 x S or N x S
        m: magnitude, N x T x F
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d} instead".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
        # N x 2F x T
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        # N x F x T
        r, i = th.chunk(c, 2, dim=1)
        m = (r**2 + i**2)**0.5
        # N x T x F
        m = th.transpose(m, 1, 2)
        return m


class MelTransform(nn.Module):
    """
    Transform linear spectrogram to mel spectrogram
    """

    def __init__(self,
                 frame_len,
                 round_pow_of_two=True,
                 sr=16000,
                 num_mels=80,
                 fmin=0.0,
                 fmax=None):
        super(MelTransform, self).__init__()
        W = init_melmat(frame_len,
                        round_pow_of_two=round_pow_of_two,
                        sr=sr,
                        num_mels=num_mels,
                        fmax=fmax,
                        fmin=fmin)
        self.num_mels, self.num_bins = W.shape
        self.W = nn.Parameter(W, requires_grad=False)
        self.fmin = fmin
        self.fmax = sr // 2 if fmax is None else fmax

    def extra_repr(self):
        return "fmin={0}, fmax={1}, weight_size={2[0]}x{2[1]}".format(
            self.fmin, self.fmax, self.W.shape)

    def forward(self, x):
        """
        x: linear spectrogram, N x T x F or N x F x T
        y: mel spectrogram, N x T x M
        """
        if x.dim() != 3:
            raise RuntimeError(
                "{} expect 3D tensor, but got {:d} instead".format(
                    self.__name__, x.dim()))
        # N x F x T => N x T x F
        if x.shape[-2] == self.num_bins:
            x = th.transpose(x, -1, -2)
        # N x T x F => N x T x M
        y = F.linear(x, self.W, bias=None)
        return y


class AsrFeature(nn.Module):
    """
    A layer to compute features for ASR task
    """

    def __init__(self,
                 feature,
                 frame_len=400,
                 frame_hop=160,
                 sr=16000,
                 window="hamm",
                 fmin=0,
                 fmax=None,
                 num_mels=80,
                 apply_log=True,
                 apply_mvn=False,
                 round_pow_of_two=True):
        super(AsrFeature, self).__init__()
        if feature not in ["mel", "spectrogram"]:
            raise RuntimeError("Unknown asr feature type: {}".format(feature))
        self.apply_mvn = None
        layers = []
        layers.append(
            Spectrogram(frame_len,
                        frame_hop,
                        window=window,
                        round_pow_of_two=round_pow_of_two))
        feature_dim = layers[-1].num_bins
        if feature == "mel":
            layers.append(
                MelTransform(frame_len,
                             round_pow_of_two=round_pow_of_two,
                             sr=sr,
                             num_mels=num_mels,
                             fmin=fmin,
                             fmax=fmax))
            feature_dim = num_mels
        if apply_mvn:
            self.apply_mvn = nn.BatchNorm1d(feature_dim)
        self.components = nn.Sequential(*layers)
        self.apply_log = apply_log
        self.dim = feature_dim

    def forward(self, x):
        """
        x: input signal, N x 1 x S or N x S
        """
        # N x
        y = self.components(x)
        if self.apply_log:
            y = th.clamp(y, min=EPSILON)
            y = th.log(y)
        if self.apply_mvn is not None:
            y = th.transpose(y, 1, 2)
            y = self.apply_mvn(y)
            y = th.transpose(y, 1, 2)
        return y