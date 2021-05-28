#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.libs import Register

PosEncodings = Register("pos_encodings")


def get_xfmr_pose(pose: str, dim: int, **kwargs) -> nn.Module:
    """
    Return position encodings layer
    Args:
        pose (str): positional encoding type, {abs|rel|xl|conv1d}
    """
    if pose not in PosEncodings:
        raise ValueError(f"Unsupported pose layer: {pose}")
    return PosEncodings[pose](dim, **kwargs)


def digit_shift(term: th.Tensor) -> th.Tensor:
    """
    Got L x N x H x S from tensor L x N x H x 2S-1
    The function is called when using 1D positional encodings instead of 2D matrices, refer testing cases in
        tests/test_function.py:test_rel_pose()
    Args:
        term (Tensor): L x N x H x 2S(L)-1
    Return:
        term (Tensor): L x N x H x S(L)
    """
    L, N, H, X = term.shape
    if L * 2 - 1 != X:
        raise RuntimeError("digit_shift: tensor shape should be: " +
                           f"L x N x H x 2L-1, but got {term.shape}")
    # L x N x H x 2L
    term_pad = tf.pad(term, (1, 0))
    # L x 2L x H x N
    term_pad = term_pad.transpose(1, -1).contiguous()
    # 2L x L x H x N
    term_pad = term_pad.view(2 * L, L, H, N)
    # L x 2L-1 x H x N
    term = term_pad[1:].view(L, 2 * L - 1, H, N)
    # L x L x H x N
    term = term[:, :L]
    # L x N x H x L
    return term.transpose(1, -1)


@PosEncodings.register("xl")
class SinPosEncoding(nn.Module):
    """
    Sinusoidals positional encoding
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0) -> None:
        super(SinPosEncoding, self).__init__()
        # D//2: 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        div_term = th.exp(-math.log(10000.0) * th.arange(0, embed_dim, 2.0) /
                          embed_dim)
        self.div_term = nn.Parameter(div_term, requires_grad=False)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sin_pos_enc(self, position: th.Tensor) -> th.Tensor:
        """
        Return sinusoidals encoding matrices
        """
        # T x D//2
        sequence = position[:, None] * self.div_term
        # T x D//2 x 2
        sin_enc = th.stack([th.sin(sequence), th.cos(sequence)], dim=-1)
        # T x D
        return sin_enc.view(position.shape[0], -1)

    def forward(self, position: th.Tensor) -> th.Tensor:
        """
        Args:
            position (Tensor): T
        Return:
            out: T x D
        """
        # T x D
        sin_enc = self._get_sin_pos_enc(position)
        # add dropout
        return self.dropout(sin_enc)


@PosEncodings.register("rel")
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

    def dumplicate(self, seq_len) -> th.Tensor:
        """
        Produce 2D matrice (dumplicated, see test_function.py:test_rel_pose)
        Args:
            seq_len (int): length of the sequence
        Return:
            encodings (Tensor): T1 x T2 x D, learnt encodings
        """
        pos_vec = th.arange(seq_len, device=self.embed.weight.device)
        rel_mat = pos_vec[None, :] - pos_vec[:, None]
        rel_mat = th.clamp(rel_mat, max=self.radius, min=-self.radius)
        return self.dropout(self.embed(rel_mat + self.radius))

    def forward(self, position: th.Tensor) -> th.Tensor:
        """
        Args:
            position (Tensor): T
        Return:
            encodings (Tensor): T x D, learnt encodings
        """
        position = th.clamp(position, max=self.radius, min=-self.radius)
        return self.dropout(self.embed(position + self.radius))


@PosEncodings.register("abs")
class InputSinPosEncoding(SinPosEncoding):
    """
    Add sinusoidals positional encodings to input features
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,
                 scaled: bool = False) -> None:
        super(InputSinPosEncoding, self).__init__(embed_dim, dropout=dropout)
        self.factor = embed_dim**0.5 if scaled else 1

    def forward(self, inp: th.Tensor, t: int = 0) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x D
        Return:
            out (Tensor): T x N x D (for transformer input)
        """
        # T
        pos = th.arange(t, t + inp.shape[1], 1.0, device=inp.device)
        # T x D
        sin_enc = self._get_sin_pos_enc(pos)
        # add dropout
        out = self.dropout(inp * self.factor + sin_enc)
        # T x N x D
        out = out.transpose(0, 1)
        return out


@PosEncodings.register("conv1d")
class Conv1dPosEncoding(nn.Module):
    """
    1D convolutional position encoding
    """

    def __init__(self,
                 embed_dim: int,
                 dropout: float = 0.1,
                 kernel: int = 33,
                 groups: int = 16):
        super(Conv1dPosEncoding, self).__init__()
        conv = nn.Conv1d(embed_dim,
                         embed_dim,
                         kernel,
                         1,
                         padding=(kernel - 1) // 2,
                         groups=groups)
        nn.init.normal_(conv.weight,
                        mean=0,
                        std=math.sqrt(4 / (kernel * embed_dim)))
        nn.init.constant_(conv.bias, 0)
        self.conv = nn.utils.weight_norm(conv, name="weight", dim=2)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x D
        Return:
            out (Tensor): T x N x D (for transformer input)
        """
        # N x D x T
        inp = inp.transpose(1, 2)
        pos = tf.gelu(self.drop(self.conv(inp)))
        return pos + inp
