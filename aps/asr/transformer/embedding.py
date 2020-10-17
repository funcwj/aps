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

    def __init__(self, embed_dim, dropout=0.1, max_len=5000, rel_enc=False):
        super(PositionalEncoding, self).__init__()
        pos_enc = th.zeros(max_len, embed_dim)
        if rel_enc:
            position = th.arange(max_len - 1, -1, -1, dtype=th.float32)
        else:
            position = th.arange(0, max_len, dtype=th.float32)
        # 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        div_term = th.exp(
            th.arange(0, embed_dim, 2, dtype=th.float32) *
            (-math.log(10000.0) / embed_dim))
        pos_enc[:, 0::2] = th.sin(position[:, None] * div_term)
        pos_enc[:, 1::2] = th.cos(position[:, None] * div_term)
        # Tmax x D
        self.register_buffer("pos_enc", pos_enc)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_scale = embed_dim**0.5
        self.rel_enc = rel_enc

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Remove "pos_enc" from state dict
        """
        state = super(PositionalEncoding,
                      self).state_dict(destination=destination,
                                       prefix=prefix,
                                       keep_vars=keep_vars)
        if "pos_enc" in state:
            state.pop("pos_enc")
        return state

    def forward(self, inp, t=0):
        """
        Args:
            inp: N x T x D
        Return:
            out: T x N x D (keep same as transformer definition)
        """
        _, T, _ = inp.shape
        # T x D
        abs_enc = self.pos_enc[t:t + T]
        # N x T x D
        inp_scale = inp * self.embed_scale
        # add dropout
        if self.rel_enc:
            out = self.dropout(inp_scale)
            abs_enc = self.dropout(abs_enc)
        else:
            out = self.dropout(inp_scale + abs_enc)
        # T x N x D
        out = out.transpose(0, 1)
        return (abs_enc, out) if self.rel_enc else out


class LinearEmbedding(nn.Module):
    """
    Linear projection embedding
    """

    def __init__(self, input_size, embed_dim=512):
        super(LinearEmbedding, self).__init__()
        self.proj = nn.Linear(input_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, inp):
        """
        Args:
            inp: features from asr transform, N x T x F
        Return:
            out: N x T x D
        """
        inp = self.norm(self.proj(inp))
        out = F.relu(inp)
        return out


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

    def forward(self, inp):
        """
        Args:
            inp: features from front-end or asr transform, N x B x T x F or N x T x F
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv1dEmbedding expect 3/4D tensor, got {inp.dim()} instead")
        if inp.dim() == 3:
            _, T, _ = inp.shape
            # N x T x F => N x F x T
            inp = inp.transpose(1, 2)
        else:
            N, _, T, _ = inp.shape
            # N x B x D x T
            inp = inp.transpose(-1, -2)
            inp = inp.contiguous()
            # N x BD x T
            inp = inp.view(N, -1, T)
        inp = F.relu(self.conv1(inp))
        out = F.relu(self.conv2(inp))
        # N x F x T/4 => N x T/4 x F
        out = out.transpose(1, 2)
        return out[:, :T // 4]


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

    def forward(self, inp):
        """
        Args:
            inp: N x T x F (from asr transform) or N x C x T x F (from front-end processing)
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv2dEmbedding expect 3/4D tensor, got {inp.dim()} instead")
        L = inp.size(-2)
        # N x 1 x T x F => N x A x T' x F'
        out = F.relu(self.conv1(inp[:, None] if inp.dim() == 3 else inp))
        # N x A x T' x F'
        out = F.relu(self.conv2(out))
        # N x T' x A x F'
        out = out.transpose(1, 2)
        N, T, _, _ = out.shape
        out = out.contiguous()
        out = out.view(N, T, -1)
        # N x T' x D
        out = self.proj(out[:, :L // 4])
        return out


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
                 other_opts=-1,
                 rel_enc=False):
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
        self.posencode = PositionalEncoding(embed_dim,
                                            dropout=dropout,
                                            rel_enc=rel_enc,
                                            max_len=6000)

    def forward(self, inp, t=0):
        """
        Args:
            inp: N x T x F (from asr transform)
        Return:
            out: T' x N x F (to feed transformer)
        """
        out = self.embed(inp)
        return self.posencode(out, t=t)
