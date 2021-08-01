#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Tuple


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


def prep_sub_mask(num_frames: int, device: th.device = "cpu") -> th.Tensor:
    """
    Prepare the square sub-sequence masks (-inf/0)
    e.g., for num_frames = 8:
    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    ones = th.ones(num_frames, num_frames, device=device)
    mask = (th.triu(ones, diagonal=1) == 1).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def prep_context_mask(num_frames: int,
                      chunk_size: int = 1,
                      lctx: int = 0,
                      rctx: int = 0,
                      device: th.device = "cpu") -> th.Tensor:
    """
    Prepare the square masks (-inf/0) for context masking
    for chunk_size = 1, lctx = -1, rctx = 0, it equals to prep_sub_mask(...)
    e.g., for chunk_size = 1, num_frames = 8, rctx = 2, lctx = 1:
    tensor([[0., 0., 0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf, -inf, -inf],
            [-inf, 0., 0., 0., 0., -inf, -inf, -inf],
            [-inf, -inf, 0., 0., 0., 0., -inf, -inf],
            [-inf, -inf, -inf, 0., 0., 0., 0., -inf],
            [-inf, -inf, -inf, -inf, 0., 0., 0., 0.],
            [-inf, -inf, -inf, -inf, -inf, 0., 0., 0.],
            [-inf, -inf, -inf, -inf, -inf, -inf, 0., 0.]])
    """
    # -1 means inf
    if lctx < 0:
        lctx = num_frames
    if rctx < 0:
        rctx = num_frames
    zeros = th.zeros(num_frames, device=device)
    index = th.arange(0, num_frames, device=device)
    index_seqs = th.repeat_interleave(index[None, ...], num_frames, 0)
    # limit right context
    right = (index // chunk_size + rctx) * chunk_size
    right_mask = index_seqs > right[..., None]
    mask = zeros.masked_fill(right_mask, float("-inf"))
    # limit left context
    left = th.clamp_min((index // chunk_size - lctx) * chunk_size, 0)
    left_mask = index_seqs < left[..., None]
    mask = mask.masked_fill(left_mask, float("-inf"))
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
