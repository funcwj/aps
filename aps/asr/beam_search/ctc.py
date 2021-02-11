#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from aps.const import NEG_INF
from typing import Optional


def fix_gamma(beam_size: int, gamma: th.Tensor, point: th.Tensor) -> th.Tensor:
    """
    Adjust gamma matrix
    Args:
        gamma: T x N*beam
        point: N
    """
    # T x N x beam_size
    T, _ = gamma.shape
    gamma = gamma.view(gamma.shape[0], -1, beam_size)
    gamma = gamma[:, point]
    return gamma.view(T, -1)


class CtcScore(object):
    """
    To compute the CTC score given decoding sequence and
    helps the beam search in attention based AM

    Args:
        ctc_prob: T x V
    """

    def __init__(self,
                 ctc_prob: th.Tensor,
                 eos: int = 1,
                 batch_size: int = 8,
                 beam_size: int = 16) -> None:
        # apply softmax
        self.ctc_prob = th.log_softmax(ctc_prob, dim=-1)
        self.T = ctc_prob.shape[0]
        self.device = ctc_prob.device
        self.eos = eos
        # blank is last symbol: see aps.conf:load_am_conf(...)
        self.blank = -1
        # eq (51) NEG_INF ~ log(0), T x N*beam
        self.gamma_n_g = th.full((self.T, batch_size * beam_size),
                                 NEG_INF,
                                 device=self.device)
        self.gamma_b_g = th.zeros(self.T,
                                  batch_size * beam_size,
                                  device=self.device)
        # eq (52)
        self.gamma_b_g[0] = self.ctc_prob[0, self.blank]
        for t in range(1, self.T):
            self.gamma_b_g[t] = (self.gamma_b_g[t - 1] +
                                 self.ctc_prob[t, self.blank])
        # ctc score in previous steps
        self.ctc_score = th.zeros(1, batch_size * beam_size, device=self.device)
        self.neg_inf = th.tensor(NEG_INF).to(self.device)

    def score(self,
              g: th.Tensor,
              c: th.Tensor,
              point: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Args:
            g (Tensor): N x U
            c (Tensor): N*beam
            point (Tensor or None): N
        Return:
            score (Tensor): N x beam
        """
        beam = c.shape[0] // g.shape[0]
        # N*beam x U
        repeat_g = th.repeat_interleave(g, beam, 0)

        if point is not None:
            self.gamma_b_g = fix_gamma(beam, self.gamma_b_g, point)
            self.gamma_n_g = fix_gamma(beam, self.gamma_n_g, point)
            self.ctc_score = fix_gamma(beam, self.ctc_score, point)
        # T x N*beam
        gamma_n_h = th.zeros_like(self.gamma_n_g)
        gamma_b_h = th.zeros_like(self.gamma_n_g)
        gamma_n_h[0] = self.ctc_prob[0, c] if g.shape[1] == 1 else NEG_INF
        gamma_b_h[0] = NEG_INF

        # N*beam
        score = gamma_n_h[0]
        repeat = repeat_g[:, -1] != c
        for t in range(1, self.T):
            # N*beam
            term = th.where(repeat, self.gamma_n_g[t - 1], self.neg_inf)
            phi = th.logaddexp(self.gamma_b_g[t - 1], term)
            gamma_n_h[t] = th.logaddexp(gamma_n_h[t - 1],
                                        phi) + self.ctc_prob[t, c]
            gamma_b_h[t] = th.logaddexp(
                gamma_b_h[t - 1], gamma_n_h[t - 1]) + self.ctc_prob[t,
                                                                    self.blank]
            score = th.logaddexp(score, phi + self.ctc_prob[t, c])
        # fix eos
        is_eos = c == self.eos
        gamma_nb_g = th.logaddexp(self.gamma_b_g, self.gamma_n_g)
        score[is_eos] = gamma_nb_g[-1, is_eos]
        delta_score = (score - self.ctc_score).view(-1, beam)

        self.gamma_n_g = gamma_n_h
        self.gamma_b_g = gamma_b_h
        self.ctc_score = score[None, ...]
        # N x beam
        return delta_score
