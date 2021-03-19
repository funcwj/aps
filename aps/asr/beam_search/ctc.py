#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from collections import defaultdict
from aps.const import NEG_INF
from aps.utils import get_logger
from typing import Dict, List, Union

logger = get_logger(__name__)


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
    # NOTE: actually do not need sos/eos here, just place it in the sentence
    prev_beam = [(str(sos), PrefixScore(zero, neg_inf))]
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
                    prefix_symb = prefix + f",{symb}"
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
        "score":
            score.score() / (1 if len_norm else len(prefix.split(",")) - 1),
        "trans":
            list(map(int, (prefix + f",{eos}").split(",")))
    } for prefix, score in prev_beam[:nbest]]


def ctc_viterbi_align(ctc_enc: th.Tensor,
                      dec_seq: th.Tensor,
                      blank: int = -1) -> Dict:
    """
    Get alignment on CTC prob using viterbi algothrim
    Args:
        ctc_enc (th.Tensor): T x V
        dec_seq (th.Tensor): U (remove eos & sos)
    Return:
        align (th.Tensor): T
    """
    if blank < 0:
        raise ValueError(f"Invalid blank ID: {blank}")

    ctc_prob = th.log_softmax(ctc_enc, -1)
    T, V = ctc_prob.shape
    logger.info(f"--- shape of the encoder output (CTC): {T} x {V}")

    U = dec_seq.shape[-1]
    if U * 2 + 1 > T:
        raise ValueError(f"Invalid target length: {U}")

    dec_seq = dec_seq.tolist()
    # T x U*2+1
    score = NEG_INF * th.ones(T, U * 2 + 1, device=ctc_prob.device)
    point = -1 * th.ones(T, U * 2 + 1, dtype=th.int32, device=ctc_prob.device)

    # time step: 0
    score[0, 0] = ctc_prob[0, blank]
    score[0, 1] = ctc_prob[0, dec_seq[0]]

    # time step: 1 -> T - 1
    for t in range(1, T):
        max_u_step = min(t * 2, U * 2 + 1)
        for u in range(max_u_step):
            # blank node
            if u % 2 == 0:
                ctc_score = ctc_prob[t, blank]
                # [u, u - 1]
                tok_index = range(u, u - 2, -1)
            # non-blank node
            else:
                u_nb = (u - 1) // 2
                ctc_score = ctc_prob[t, u_nb]
                if u_nb != 0 and dec_seq[u_nb] == dec_seq[u_nb - 1]:
                    # [u, u - 1]
                    tok_index = range(u, u - 2, -1)
                else:
                    # [u, u - 1, u - 2]
                    tok_index = range(u, u - 3, -1)
            prev_score = th.stack([score[t - 1, u] for u in tok_index])
            best_score, best_index = th.max(prev_score, 0)
            score[t, u] = best_score + ctc_score
            # point to time t - 1
            point[t, u] = u - best_index.item()

    align_score = None
    align = []
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            last_score = th.stack([score[-1, u] for u in [-1, -2]])
            best_score, best_index = th.max(last_score, 0)
            align.append(U * 2 - best_index.item())
            align_score = best_score.item()
        else:
            align.append(point[t + 1, align[-1]].item())
    align = align[::-1]
    for t in range(T):
        align[t] = blank if align[t] % 2 == 0 else dec_seq[(align[t] - 1) // 2]
    align_check = [a for a in align if a != blank]
    # check alignment
    assert sum([a != b for a, b in zip(align_check, dec_seq)]) == 0

    def ali_map(v):
        return "*" if v == blank else str(v)

    return {
        "score": align_score,
        "align_seq": align,
        "align_str": " ".join(map(ali_map, align))
    }


class CtcScorer(nn.Module):
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
                 beam_size: int = 12) -> None:
        super(CtcScorer, self).__init__()
        # apply softmax
        self.ctc_prob = th.log_softmax(ctc_prob, dim=-1)
        T, V = self.ctc_prob.shape
        logger.info(f"--- shape of the encoder output (CTC): {T} x {V}")
        self.T = T
        self.device = ctc_prob.device
        self.eos = eos
        # blank is last symbol: see aps.conf:load_am_conf(...)
        self.blank = -1
        self.offset = th.arange(batch_size, device=self.device)
        self.beam_size = beam_size
        # eq (51) NEG_INF ~ log(0), T x N
        self.gamma_n_g = th.full((self.T, batch_size),
                                 NEG_INF,
                                 device=self.device)
        self.gamma_b_g = th.zeros(self.T, batch_size, device=self.device)
        # eq (52)
        self.gamma_b_g[0] = self.ctc_prob[0, self.blank]
        for t in range(1, self.T):
            self.gamma_b_g[t] = (self.gamma_b_g[t - 1] +
                                 self.ctc_prob[t, self.blank])
        # ctc score in previous steps
        self.ctc_score = th.zeros(1, batch_size, device=self.device)
        self.neg_inf = th.tensor(NEG_INF).to(self.device)

    def update_var(self, point: Union[th.Tensor, int]) -> None:
        """
        Args:
            point (Tensor or int): N x att_beam or att_beam
        """
        if isinstance(point, th.Tensor):
            assert point.dim() in [1, 2]
            if point.dim() == 2:
                point = (point + self.offset[:, None] * self.beam_size).view(-1)
        else:
            point = self.offset + point
        self.ctc_score = self.ctc_score[:, point]
        self.gamma_b_g = self.gamma_b_g[:, point]
        self.gamma_n_g = self.gamma_n_g[:, point]

    def forward(self, g: th.Tensor, c: th.Tensor) -> th.Tensor:
        """
        Args:
            g (Tensor): N x U
            c (Tensor): N*ctc_beam
        Return:
            score (Tensor): N x ctc_beam
        """
        # CTC beam
        ctc_beam = c.shape[0] // g.shape[0]
        # N*ctc_beam x U
        repeat_g = th.repeat_interleave(g, ctc_beam, 0)

        # 1 x N*ctc_beam
        self.ctc_score = th.repeat_interleave(self.ctc_score, ctc_beam, -1)
        # T x N*ctc_beam
        self.gamma_n_g = th.repeat_interleave(self.gamma_n_g, ctc_beam, -1)
        self.gamma_b_g = th.repeat_interleave(self.gamma_b_g, ctc_beam, -1)
        # T x N*ctc_beam
        gamma_n_h = th.zeros_like(self.gamma_n_g)
        gamma_b_h = th.zeros_like(self.gamma_n_g)
        # zero based
        glen = g.shape[-1] - 1
        start = max(glen, 1)
        gamma_n_h[start - 1] = self.ctc_prob[0, c] if glen == 0 else NEG_INF
        gamma_b_h[start - 1] = NEG_INF

        # N*ctc_beam
        score = gamma_n_h[start - 1]
        repeat = repeat_g[:, -1] != c
        for t in range(start, self.T):
            # N*ctc_beam
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
        gamma_nb_g = th.logaddexp(self.gamma_b_g[-1], self.gamma_n_g[-1])
        score[is_eos] = gamma_nb_g[is_eos]
        delta_score = (score - self.ctc_score).view(-1, ctc_beam)

        self.gamma_n_g = gamma_n_h
        self.gamma_b_g = gamma_b_h
        self.ctc_score = score[None, ...]
        # N x ctc_beam
        return delta_score
