#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from collections import defaultdict
from aps.const import NEG_INF
from aps.utils import get_logger
from typing import Optional, Dict, List

logger = get_logger(__name__)


def fix_gamma(beam_size: int, gamma: th.Tensor, point: th.Tensor) -> th.Tensor:
    """
    Adjust gamma matrix
    Args:
        gamma: T x N*beam
        point: N
    """
    # T x N x beam_size
    gamma = gamma.view(gamma.shape[0], -1, beam_size)
    gamma = gamma[:, point]
    return gamma.view(gamma.shape[0], -1)


class PrefixScore(object):
    """
    CTC prefix score used for beam search
    """

    def __init__(self, log_pb: th.Tensor, log_pn: th.Tensor) -> None:
        # score end with blank
        self.log_pb = log_pb
        # score end with non-blank
        self.log_pn = log_pn

    def score(self) -> th.Tensor:
        return th.logaddexp(self.log_pb, self.log_pn)


def ctc_beam_search(ctc_prob: th.Tensor,
                    beam_size: int = 8,
                    blank: int = -1,
                    nbest: int = 1,
                    sos: int = -1,
                    eos: int = -1,
                    len_norm: bool = True,
                    **kwargs) -> List[Dict]:
    """
    Do CTC prefix beam search
    Args:
        ctc_prob: T x V
    """
    if sos < 0 or eos < 0:
        raise ValueError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if blank < 0:
        raise ValueError(f"Invalid blank ID: {blank}")
    ctc_prob = th.log_softmax(ctc_prob, -1)
    # T x B
    topk_score, topk_token = th.topk(ctc_prob, beam_size, -1)
    T, V = ctc_prob.shape
    logger.info(f"--- shape of the encoder output (CTC): {T} x {V}")
    neg_inf = th.tensor(NEG_INF).to(ctc_prob.device)
    zero = th.tensor(0.0).to(ctc_prob.device)
    # (prefix, log_pb, log_pn)
    prev_beam = [((sos,), PrefixScore(zero, neg_inf))]
    for t in range(T):
        next_beam = defaultdict(lambda: PrefixScore(neg_inf, neg_inf))
        for n in range(beam_size):
            symb = topk_token[t, n].item()
            logp = topk_score[t, n].item()

            for prefix, prev in prev_beam[:beam_size]:
                # update log_pb only
                if symb == blank:
                    other = next_beam[prefix]
                    log_pb_update = th.logaddexp(prev.score() + logp,
                                                 other.log_pb)
                    next_beam[prefix] = PrefixScore(log_pb_update, other.log_pn)
                else:
                    prefix_symb = prefix + (symb,)
                    other = next_beam[prefix_symb]
                    # repeat
                    if prefix[-1] == symb:
                        log_pn_update = th.logaddexp(prev.log_pb + logp,
                                                     other.log_pn)
                    else:
                        log_pn_update = th.logaddexp(prev.score() + logp,
                                                     other.log_pn)
                    # update log_pn only
                    next_beam[prefix_symb] = PrefixScore(
                        other.log_pb, log_pn_update)
                    # repeat case
                    if prefix[-1] == symb:
                        other = next_beam[prefix]
                        log_pn_update = th.logaddexp(prev.log_pn + logp,
                                                     other.log_pn)
                        next_beam[prefix] = PrefixScore(other.log_pb,
                                                        log_pn_update)
        # keep top-#beam
        prev_beam = sorted(next_beam.items(),
                           key=lambda n: n[1].score(),
                           reverse=True)
    return [{
        "score": score.score() / (1 if len_norm else len(prefix) - 1),
        "trans": prefix + (eos,)
    } for prefix, score in prev_beam[:nbest]]


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
        T, V = self.ctc_prob.shape
        logger.info(f"--- shape of the encoder output (CTC): {T} x {V}")
        self.T = T
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
        # zero based
        glen = g.shape[-1] - 1
        start = max(glen, 1)
        gamma_n_h[start - 1] = self.ctc_prob[0, c] if glen == 0 else NEG_INF
        gamma_b_h[start - 1] = NEG_INF

        # N*beam
        score = gamma_n_h[start - 1]
        repeat = repeat_g[:, -1] != c
        for t in range(start, self.T):
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
