#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Union, Tuple

HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]


def adjust_hidden(back_point: th.Tensor, state: HiddenType) -> HiddenType:
    """
    Adjust RNN hidden states
    Args:
        back_point (Tensor): N
        state (None or Tensor, [Tensor, Tensor])
    Return:
        state (None or Tensor, [Tensor, Tensor])
    """
    if state is not None:
        if isinstance(state, tuple):
            # shape: num_layers * num_directions, batch, hidden_size
            h, c = state
            state = (h[:, back_point], c[:, back_point])
        else:
            state = state[:, back_point]
    return state


def lm_score(lm: nn.Module, back_point: th.Tensor, prev_token: th.Tensor,
             state: HiddenType) -> Tuple[th.Tensor, HiddenType]:
    """
    Get LM prob/score
    Args:
        back_point (Tensor): N
        prev_token (Tensor): N
        state (HiddenType): hidden state from previous step
    Return:
        score (Tensor): beam x V
        state (HiddenType): new hidden state
    """
    # adjust order
    state = adjust_hidden(back_point, state)
    # LM
    lmout, state = lm(prev_token[..., None], state)
    # beam x V
    score = tf.log_softmax(lmout[:, 0], dim=-1)
    # return state & score
    return (score, state)
