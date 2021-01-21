#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Union, Tuple
from aps.asr.lm.ngram import NgramLM

HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]
LmType = Union[nn.Module, NgramLM]


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


def ngram_score(lm: NgramLM, back_point: th.Tensor, prev_token: th.Tensor,
                state):
    """
    Get ngram LM score
    Args:
        back_point (Tensor): N
        state (list[list(State)]): ngram LM states
    Return:
        score (Tensor): beam x V
        state (list[list(State)]): new LM state
    """
    if state is None:
        prev_state = None
    else:
        # adjust states
        ptr = back_point.tolist()
        prev_state = [state[p] for p in ptr]
    return lm(prev_token, prev_state)


def rnnlm_score(rnnlm: nn.Module, back_point: th.Tensor, prev_token: th.Tensor,
                state: HiddenType) -> Tuple[th.Tensor, HiddenType]:
    """
    Get RNN LM prob/score
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
    lmout, state = rnnlm(prev_token[..., None], state)
    # beam x V
    score = tf.log_softmax(lmout[:, 0], dim=-1)
    # return state & score
    return (score, state)


def lm_score_impl(lm: LmType, back_point: th.Tensor, prev_token: th.Tensor,
                  state):
    """
    Get ngram/rnnlm score (wraps {rnnlm|ngram}_score functions)
    """
    if isinstance(lm, nn.Module):
        return rnnlm_score(lm, back_point, prev_token, state)
    elif isinstance(lm, NgramLM):
        return ngram_score(lm, back_point, prev_token, state)
    else:
        raise TypeError(f"Unsupported LM type: {type(lm)}")
