#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import torch as th
import torch.nn as nn

from aps.libs import Register

PosEncodings = Register("pos_encodings")


def get_xfmr_pose(pose_name: str, embed_dim: int, **kwargs) -> nn.Module:
    """
    Return position encodings layer
    """
    if pose_name not in PosEncodings:
        raise ValueError(f"Unsupported position encoding layer: {pose_name}")
    return PosEncodings[pose_name](embed_dim, **kwargs)


@PosEncodings.register("sin")
class SinPosEncoding(nn.Module):
    """
    Sinusoidals positional encoding
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super(SinPosEncoding, self).__init__()
        # D//2: 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0) / embed_dim))
        self.div_term = nn.Parameter(th.exp(
            th.arange(0, embed_dim, 2.0) * (-math.log(10000.0) / embed_dim)),
                                     requires_grad=False)
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
            position (Tensor): N
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

    def forward(self, len1: int, len2: int) -> th.Tensor:
        """
        Args:
            len1 (int): length of the sequence1
            len2 (int): length of the sequence2
        Return:
            encodings (Tensor): T1 x T2 x D, learnt encodings
        """
        pos_vec = th.arange(max(len1, len2), device=self.embed.weight.device)
        rel_mat = pos_vec[:len1, None] - pos_vec[None, :len2]
        rel_mat = th.clamp(rel_mat, max=self.radius, min=-self.radius)
        return self.dropout(self.embed(rel_mat + self.radius))


@PosEncodings.register("inp_sin")
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
            inp (Tensor): N x T x D
        Return:
            out (Tensor): N x T x D
        """
        # T
        pos = th.arange(t, t + inp.shape[1], 1.0, device=inp.device)
        # T x D
        sin_enc = self._get_sin_pos_enc(pos)
        # add dropout
        out = self.dropout(inp * self.embed_scale + sin_enc)
        # T x N x D
        out = out.transpose(0, 1)
        return out
