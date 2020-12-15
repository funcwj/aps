#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import torch as th
import torch.nn as nn


class SinPosEncoding(nn.Module):
    """
    Sinusoidals positional encoding
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super(SinPosEncoding, self).__init__()
        # D//2: 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        div_term = th.exp(
            th.arange(0, embed_dim, 2.0) * (-math.log(10000.0) / embed_dim))
        self.register_buffer("div_term", div_term)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sin_pos_enc(self, num_frames: int, base: int = 0) -> th.Tensor:
        """
        Return sinusoidals encoding matrices
        """
        # T
        sequence = th.arange(base, base + num_frames, 1.0)
        # T x D//2
        sequence = sequence[:, None] * self.div_term
        # T x D//2 x 2
        sin_enc = th.stack([th.sin(sequence), th.cos(sequence)], dim=-1)
        # T x D
        return sin_enc.view(num_frames, -1)

    def forward(self, length: int, t: int = 0) -> th.Tensor:
        """
        Args:
            length (int): length of sequence
        Return:
            out: T x D
        """
        # T x D
        sin_enc = self._get_sin_pos_enc(length, base=t)
        # add dropout
        return self.dropout(sin_enc)


class RelPosEncoding(nn.Module):
    """
    Relative positional encoding
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,
                 radius: int = 128) -> None:
        super(RelPosEncoding, self).__init__()
        self.radius = radius
        self.embed = nn.Embedding(radius * 2 + 1, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, num_frames: int) -> th.Tensor:
        """
        Args:
            num_frames (int): length of the sequence
        Return:
            enc (Tensor): T x T x D, learnt encodings
        """
        # T => T x T
        vec = th.arange(num_frames, device=self.embed.weight.device)
        mat = th.clamp(vec[:, None] - vec[None, :],
                       max=self.radius,
                       min=-self.radius)
        return self.dropout(self.embed(mat + self.radius))


class InputSinPosEncoding(SinPosEncoding):
    """
    Add sinusoidals positional encodings to input features
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,
                 scale_embed: bool = False) -> None:
        super(InputSinPosEncoding, self).__init__(embed_dim, dropout=dropout)
        self.embed_scale = embed_dim**0.5 if scale_embed else 1

    def forward(self, inp: th.Tensor, t: int = 0) -> th.Tensor:
        """
        Args:
            inp: N x T x D
        Return:
            out: N x T x D
        """
        # T x D
        sin_enc = self._get_sin_pos_enc(inp.shape[1], base=t)
        # add dropout
        out = self.dropout(inp * self.embed_scale + sin_enc)
        # T x N x D
        out = out.transpose(0, 1)
        return out
