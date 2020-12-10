#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for CTC
"""
import torch as th
import torch.nn.functional as tf

from typing import List, Dict


def greedy_search(enc_out: th.Tensor, blank: int = -1) -> List[Dict]:
    """
    Greedy search
    Args:
        enc_out (Tensor): 1 x T x V
    """
    N, T, _ = enc_out.shape
    if blank < 0:
        raise ValueError(f"Invalid blank id: {blank}")
    if N != 1:
        raise ValueError(
            f"Got batch size {N:d}, now only support one utterance")
    # T x V
    prob = tf.log_softmax(enc_out[0], -1)
    # T x 1
    best_score, best_token = th.topk(prob, 1, dim=-1)
    # process tokens
    best_token = best_token[:, 0].tolist()
    token = [blank]
    for t in range(T):
        if token[-1] != best_token[t]:
            token.append(best_token[t])
    return [{
        "score": th.sum(best_score).item(),
        # remove blank
        "trans": [t for t in token if t != blank]
    }]


def beam_search(enc_out: th.Tensor,
                beam_size: int,
                blank: int = 0) -> List[Dict]:
    """
    Implementation of CTC prefix beam search
    """
    pass
