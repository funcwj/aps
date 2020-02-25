# wujian@2019

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import librosa.filters as filters

from scipy.fftpack import dct
from kaldi_python_io.functional import read_kaldi_mat

EPSILON = th.finfo(th.float32).eps


def init_window(wnd, frame_len):
    """
    Return window coefficient
    """
    def sqrthann(frame_len):
        return th.hann_window(frame_len)**0.5

    if wnd not in ["sqrthann", "hann", "hamm", "blackman"]:
        raise RuntimeError(f"Unknown window type: {wnd}")

    wnd_tpl = {
        "sqrthann": sqrthann,
        "hann": th.hann_window,
        "hamm": th.hamming_window,
        "blackman": th.blackman_window,
        "bartlett": th.bartlett_window
    }
    c = wnd_tpl[wnd](frame_len)
    return c


def init_kernel(frame_len, frame_hop, round_pow_of_two=True,
                window="sqrthann"):
    """
    Return STFT kernels
    """
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # window
    W = init_window(window, frame_len)
    # scale factor to make same magnitude after iSTFT
    if window == "sqrthann":
        S = 0.5 * (N * N / frame_hop)**0.5
    else:
        S = 1
    # F x N/2+1 x 2
    K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


def init_melfilter(frame_len,
                   round_pow_of_two=True,
                   sr=16000,
                   num_mels=80,
                   fmin=0.0,
                   fmax=None):
    """
    Return mel-filters
    """
    # FFT points
    N = 2**math.ceil(math.log2(frame_len)) if round_pow_of_two else frame_len
    # fmin & fmax
    fmax = sr // 2 if fmax is None else min(fmax, sr // 2)
    # mel-matrix
    mel = filters.mel(sr, N, n_mels=num_mels, fmax=fmax, fmin=fmin, htk=True)
    # num_mels x (N // 2 + 1)
    return th.tensor(mel, dtype=th.float32)


def init_dct(num_ceps=13, num_mels=40):
    """
    Return DCT matrix
    """
    dct_mat = dct(np.eye(num_mels), norm="ortho")[:num_ceps]
    # num_ceps x num_mels
    return th.tensor(dct_mat, dtype=th.float32)


def load_gcmvn_stats(cmvn_mat):
    """
    Compute mean/std from Kaldi's cmvn.mat
    """
    cmvn = th.tensor(read_kaldi_mat(cmvn_mat), dtype=th.float32)
    N = cmvn[0, -1]
    mean = cmvn[0, :-1] / N
    var = cmvn[1, :-1] / N - mean**2
    return mean, var**0.5

class STFTBase(nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 round_pow_of_two=True):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len,
                        frame_hop,
                        round_pow_of_two=round_pow_of_two,
                        window=window)
        self.K = nn.Parameter(K, requires_grad=False)
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.window = window
        self.num_bins = self.K.shape[0] // 2

    def num_frames(self, num_samples):
        if th.sum(num_samples <= self.frame_len):
            raise RuntimeError(f"Audio samples {num_samples.cpu()} less " +
                               f"than frame_len ({self.frame_len})")
        return (num_samples - self.frame_len) // self.frame_hop + 1

    def extra_repr(self):
        return (f"window={self.window}, stride={self.frame_hop}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}")


class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__class__.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = F.conv1d(x, self.K, stride=self.frame_hop, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.view(N * C, 1, S)
            # NC x 2F x T
            c = F.conv1d(x, self.K, stride=self.frame_hop, padding=0)
            # N x C x 2F x T
            c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
        if cplx:
            return (r, i)
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return (m, p)


class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.frame_hop, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s