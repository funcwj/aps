#!/usr/bin/env python

# wujian@2020

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Reference: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos_enc = th.zeros(max_len, embed_dim)
        position = th.arange(0, max_len, dtype=th.float32)
        div_term = th.exp(
            th.arange(0, embed_dim, 2, dtype=th.float32) *
            (-math.log(10000.0) / embed_dim))
        pos_enc[:, 0::2] = th.sin(position[:, None] * div_term)
        pos_enc[:, 1::2] = th.cos(position[:, None] * div_term)
        # Tmax x 1 x D
        self.register_buffer("pos_enc", pos_enc[:, None])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, t=0):
        """
        args:
            x: N x T x D 
        return:
            y: T x N x D (keep same as transformer definition)
        """
        _, T, _ = x.shape
        # T x N x D
        x = x.transpose(0, 1)
        # Tmax x 1 x D
        x = x + self.pos_enc[t:t + T, :]
        x = self.dropout(x)
        return x


class LinearEmbedding(nn.Module):
    """
    Linear projection embedding
    """
    def __init__(self, input_size, embed_dim=512):
        super(LinearEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        args:
            x: N x T x F (from asr transform)
        """
        x = self.norm(self.proj(x))
        x = F.relu(x)
        return x


class Conv1dEmbedding(nn.Module):
    """
    1d-conv embedding
    """
    def __init__(self, input_size, embed_dim=512, inner_channels=256):
        super(Conv1dEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(input_size,
                               inner_channels,
                               3,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv1d(inner_channels,
                               embed_dim,
                               3,
                               stride=2,
                               padding=1)

    def forward(self, x):
        """
        args:
            x: N x B x T x F or N x T x F (from front-end or asr transform)
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv1dEmbedding expect 3/4D tensor, got {x.dim()} instead")
        if x.dim() == 3:
            _, T, _ = x.shape
            # N x T x F => N x F x T
            x = x.transpose(1, 2)
        else:
            N, _, T, _ = x.shape
            # N x B x D x T
            x = x.transpose(-1, -2)
            x = x.contiguous()
            # N x BD x T
            x = x.view(N, -1, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # N x F x T/4 => N x T/4 x F
        x = x.transpose(1, 2)
        return x[:, :T // 4]


class Conv2dEmbedding(nn.Module):
    """
    2d-conv embedding described in:
    Speech-transformer: A no-recurrence sequence-to-sequence model for speech recognition
    """
    def __init__(self, input_size, embed_dim=512, input_channels=1):
        super(Conv2dEmbedding, self).__init__()
        self.conv1 = nn.Conv2d(input_channels,
                               embed_dim,
                               3,
                               stride=2,
                               padding=1)
        input_size = (input_size - 1) // 2 + 1
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1)
        input_size = (input_size - 1) // 2 + 1
        self.proj = nn.Linear(input_size * embed_dim, embed_dim)

    def forward(self, x):
        """
        args:
            x: N x T x F (from asr transform) or 
               N x C x T x F (from front-end processing)
        """
        if x.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv2dEmbedding expect 3/4D tensor, got {x.dim()} instead")
        L = x.size(-2)
        # N x 1 x T x F => N x A x T' x F'
        x = F.relu(self.conv1(x[:, None] if x.dim() == 3 else x))
        # N x A x T' x F'
        x = F.relu(self.conv2(x))
        # N x T' x A x F'
        x = x.transpose(1, 2)
        N, T, _, _ = x.shape
        x = x.contiguous()
        x = x.view(N, T, -1)
        # N x T' x D
        x = self.proj(x[:, :L // 4])
        return x


class IOEmbedding(nn.Module):
    """
    Kinds of feature embedding layer for ASR tasks
        1) Linear transform
        2) Conv1d transform
        3) Conv2d transform
        4) Sparse transform
    """
    def __init__(self,
                 embed_type,
                 feature_dim,
                 embed_dim=512,
                 dropout=0.1,
                 other_opts=-1):
        super(IOEmbedding, self).__init__()
        if embed_type == "linear":
            self.embed = LinearEmbedding(feature_dim, embed_dim=embed_dim)
        elif embed_type == "conv2d":
            self.embed = Conv2dEmbedding(
                feature_dim,
                embed_dim=embed_dim,
                input_channels=1 if other_opts <= 0 else other_opts)
        elif embed_type == "conv1d":
            self.embed = Conv1dEmbedding(
                feature_dim,
                embed_dim=embed_dim,
                inner_channels=embed_dim if other_opts <= 0 else other_opts)
        elif embed_type == "sparse":
            self.embed = nn.Embedding(feature_dim, embed_dim)
        else:
            raise RuntimeError(f"Unsupported embedding type: {embed_type}")
        self.posencode = PositionalEncoding(embed_dim, dropout=dropout)

    def forward(self, x, t=0):
        """
        args:
            x: N x T x F (from asr transform)
        return:
            y: T' x N x F (to feed transformer)
        """
        y = self.embed(x)
        y = self.posencode(y, t=t)
        return y