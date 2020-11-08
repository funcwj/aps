#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# From https://github.com/funcwj/voice-filter/tree/master/nnet

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Conv2dBlock(nn.Module):
    """
    2D convolutional blocks used in VoiceFilter
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(5, 5),
                 dilation=(1, 1)):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=dilation,
            padding=tuple(
                d * (k - 1) // 2 for k, d in zip(kernel_size, dilation)))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: N x F x T
        """
        x = self.bn(self.conv(x))
        return F.relu(x)


class VoiceFilter(nn.Module):
    """
    Reference from
    VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking
    """

    def __init__(self,
                 frame_len,
                 frame_hop,
                 enh_transform,
                 num_bins=257,
                 embedding_dim=512,
                 lstm_dim=400,
                 linear_dim=600,
                 l2_norm=True,
                 bidirectional=False,
                 non_linear="relu"):
        super(VoiceFilter, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError(
                "Unsupported non-linear function: {}".format(non_linear))
        self.enh_transform = enh_transform
        self.cnn_f = Conv2dBlock(1, 64, kernel_size=(7, 1))
        self.cnn_t = Conv2dBlock(64, 64, kernel_size=(1, 7))
        blocks = []
        for d in range(5):
            blocks.append(
                Conv2dBlock(64, 64, kernel_size=(5, 5), dilation=(1, 2**d)))
        self.cnn_tf = nn.Sequential(*blocks)
        self.proj = Conv2dBlock(64, 8, kernel_size=(1, 1))
        self.lstm = nn.LSTM(8 * num_bins + embedding_dim,
                            lstm_dim,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.mask = nn.Sequential(
            nn.Linear(lstm_dim * 2 if bidirectional else lstm_dim, linear_dim),
            nn.ReLU(), nn.Linear(linear_dim, num_bins))
        self.non_linear = supported_nonlinear[non_linear]
        self.embedding_dim = embedding_dim
        self.l2_norm = l2_norm

    def check_args(self, x, e):
        if x.dim() != e.dim():
            raise RuntimeError(
                f"VoiceFilter got invalid input dim: x/e = {x.dim()}/{e.dim()}")
        if e.size(-1) != self.embedding_dim:
            raise RuntimeError(
                "input embedding dim do not match with " +
                f"network's, {e.size(-1)} vs {self.embedding_dim}")

    def forward(self, x, e, return_mask=False):
        """
        x: N x S
        e: N x D
        """
        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
            e = th.unsqueeze(e, 0)
        if self.l2_norm:
            e = e / th.norm(e, 2, dim=1, keepdim=True)

        # feats: N x T x F
        feats, stft, _ = self.enh_transform(x, None)
        # N x F x T
        feats = feats.transpose(1, 2)
        N, _, T = feats.shape
        # N x 1 x F x T
        y = th.unsqueeze(feats, 1)
        # N x D => N x D x T
        e = th.unsqueeze(e, 2).repeat(1, 1, T)

        y = self.cnn_f(y)
        y = self.cnn_t(y)
        y = self.cnn_tf(y)
        # N x C x F x T
        y = self.proj(y)
        # N x CF x T
        y = y.view(N, -1, T)
        # N x (CF+D) x T
        f = th.cat([y, e], 1)
        # N x T x (CF+D)
        f = th.transpose(f, 1, 2)
        f, _ = self.lstm(f)
        # N x T x F
        m = self.non_linear(self.mask(f))
        if return_mask:
            return m
        # N x F x T
        m = th.transpose(m, 1, 2)
        enh = stft * m
        # N x S
        s = self.enh_transform.inverse_stft((enh.real, enh.imag),
                                            input="complex")
        return s
