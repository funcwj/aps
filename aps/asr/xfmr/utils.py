#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Union, Tuple


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


def prep_sub_mask(T: int, device: th.device = "cpu") -> th.Tensor:
    """
    Prepare a square sub-sequence masks (-inf/0)
    egs: for N = 8, output
    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    mask = (th.triu(th.ones(T, T, device=device), diagonal=1) == 1).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class Swish(nn.Module):
    """
    Swish activation
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return inp * th.sigmoid(inp)


def get_activation_fn(activation: str) -> nn.Module:
    """
    Return activation function for self-attention
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "swish":
        return Swish()
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def get_relative_uv(shape: Tuple[int],
                    init: str = "xavier",
                    std: float = 0.02) -> nn.Parameter:
    """
    Return rel_{u,v} used in transformer-XL's MHSA
    """
    if init not in ["xavier", "normal"]:
        raise ValueError(f"Unknown init method: {init}")
    rel_mat = nn.Parameter(th.Tensor(*shape))
    if init == "xavier":
        nn.init.xavier_uniform_(rel_mat)
    if init == "uniform":
        nn.init.normal_(rel_mat, std=std)
    return rel_mat
