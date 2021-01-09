#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional, NoReturn
from aps.const import NEG_INF


@dataclass
class BeamSearchParam(object):
    """
    Parameters used in beam search
    """
    beam_size: int = 8
    batch_size: Optional[int] = None
    sos: int = 1
    eos: int = 2
    min_len: int = 1
    lm_weight: float = 0
    eos_threshold: float = 0
    device: Union[th.device, str] = "cpu"
    penalty: float = 0
    coverage: float = 0
    len_norm: bool = True


class BaseBeamTracker(object):
    """
    Base class (to be inheried)
    """

    def __init__(self, param: BeamSearchParam) -> None:
        self.param = param

    def beam_select(self, am_prob: th.Tensor, lm_prob: Union[th.Tensor, float],
                    att_ali: Optional[th.Tensor]) -> Tuple[th.Tensor]:
        """
        Perform beam selection
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        Return:
            topk_score (Tensor): N x K, topk score
            topk_token (Tensor): N x K, topk token ID
        """
        if att_ali is None:
            cov = 0
        else:
            cov = th.log(th.clamp_max(th.sum(att_ali, -1, keepdim=True), 0.5))
        # local pruning: N*beam x beam
        topk_score, topk_token = th.topk(am_prob +
                                         self.param.lm_weight * lm_prob +
                                         self.param.coverage * cov,
                                         self.param.beam_size,
                                         dim=-1)
        return (topk_score, topk_token)

    def trace_hypos(self,
                    score: List[float],
                    point: List[th.Tensor],
                    token: List[th.Tensor],
                    final: bool = False) -> List[Dict]:
        """
        Traceback decoding hypothesis
        Args:
            score (list[float]): final decoding score
            point (list[Tensor]): traceback point
            token (list[Tensor]): token sequence
            final (bool): is final step or not
        """
        trans = []
        for ptr, tok in zip(point[::-1], token[::-1]):
            trans.append(tok[point].tolist())
            point = ptr[point]
        hypos = []
        trans = trans[::-1]
        for i, s in enumerate(score):
            token = [t[i] for t in trans]
            token_len = len(token) if final else len(token) - 1
            score = s + token_len * self.param.penalty
            hypos.append({
                "score": score / (token_len if self.param.len_norm else 1),
                "trans": token + [self.param.eos] if final else token
            })
        return hypos


class BeamTracker(BaseBeamTracker):
    """
    A data structure used in beam search algothrim
    """

    def __init__(self, param: BeamSearchParam) -> None:
        super(BeamTracker, self).__init__(param)
        self.token = [
            th.tensor([param.sos] * param.beam_size, device=param.device)
        ]
        self.point = [th.tensor(range(param.beam_size), device=param.device)]
        self.score = th.zeros(param.beam_size, device=param.device)

    def __getitem__(self, t: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Return the token and backward point
        """
        return (self.token[t], self.point[t])

    def trace_back(self, final: bool = False) -> Optional[List[Dict]]:
        """
        Return decoding hypothesis
        Args:
            final (bool): is final step or not
        """
        end_eos = (self.token[-1] == self.param.eos).tolist()
        hyp = None
        if not final and sum(end_eos):
            idx = [i for i, end_with_eos in enumerate(end_eos) if end_with_eos]
            idx = th.tensor(idx, device=self.param.device)
            hyp = self._trace_back_hypos(idx, final=False)
        not_end = [not f for f in end_eos]
        if final and sum(not_end):
            idx = th.tensor([i for i, go_on in enumerate(not_end) if go_on],
                            device=self.param.device)
            hyp = self._trace_back_hypos(idx, final=True)
        # filter short utterances
        return [h for h in hyp if len(h["trans"]) >= self.param.min_len + 2]

    def prune_beam(self,
                   am_prob: th.Tensor,
                   lm_prob: Union[th.Tensor, float],
                   att_ali: Optional[th.Tensor] = None) -> NoReturn:
        """
        Prune and update score & token & backward point
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        """
        # local pruning: beam x V => beam x beam
        topk_score, topk_token = self.beam_select(am_prob, lm_prob, att_ali)

        if len(self.point) == 1:
            self.score += topk_score[0]
            self.token.append(topk_token[0])
            self.point.append(self.point[-1])
        else:
            # beam*beam
            accu_score = (self.score[..., None] + topk_score).view(-1)
            # beam*beam => beam
            self.score, topk_index = th.topk(accu_score,
                                             self.param.beam_size,
                                             dim=-1)
            # point to father's node
            self.point.append(topk_index // self.param.beam_size)
            self.token.append(topk_token.view(-1)[topk_index])

    def _trace_back_hypos(self,
                          point: th.Tensor,
                          final: bool = False) -> List[Dict]:
        """
        Trace back the decoding transcription sequence from the current time point
        Args:
            point (Tensor): initial backward point
        """
        score = self.score[point].tolist()
        self.score[point] = NEG_INF
        return self.trace_hypos(score, self.point, self.token, final=final)


class BatchBeamTracker(BaseBeamTracker):
    """
    A data structure used in batch version of the beam search
    """

    def __init__(self, batch_size: int, param: BeamSearchParam) -> None:
        super(BatchBeamTracker, self).__init__(param)
        self.param = param
        self.token = [
            th.tensor([[param.sos] * param.beam_size] * batch_size,
                      device=param.device)
        ]
        self.point = [
            th.tensor([list(range(param.beam_size))] * batch_size,
                      device=param.device)
        ]
        self.score = th.zeros(batch_size, param.beam_size, device=param.device)
        self.step_point = th.arange(0,
                                    param.beam_size * batch_size,
                                    param.beam_size,
                                    device=param.device)

    def __getitem__(self, t: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Return the token and backward point
        """
        point = self.point[t] + self.step_point[:, None]
        token = self.token[t]
        return (token.view(-1), point.view(-1))

    def trace_back(self, batch, final: bool = False) -> Optional[List[Dict]]:
        """
        Return end flags
        """
        end_eos = (self.token[-1][batch] == self.param.eos).tolist()
        hyp = None
        if not final and sum(end_eos):
            idx = [i for i, end_with_eos in enumerate(end_eos) if end_with_eos]
            idx = th.tensor(idx, device=self.score.device)
            hyp = self._trace_back_hypos(batch, idx, final=False)
        not_end = [not f for f in end_eos]
        if final and sum(not_end):
            idx = th.tensor([i for i, go_on in enumerate(not_end) if go_on],
                            device=self.score.device)
            hyp = self._trace_back_hypos(batch, idx, final=True)
        # filter short utterances
        return [h for h in hyp if len(h["trans"]) >= self.param.min_len + 2]

    def prune_beam(self,
                   am_prob: th.Tensor,
                   lm_prob: Union[th.Tensor, float],
                   att_ali: Optional[th.Tensor] = None) -> NoReturn:
        """
        Prune and update score & token & backward point
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        """
        # local pruning: beam x V => beam x beam
        topk_score, topk_token = self.beam_select(am_prob, lm_prob, att_ali)
        if len(self.point) == 1:
            # N x beam
            self.score += topk_score[::self.param.beam_size]
            self.point.append(self.point[-1])
            token = topk_token[::self.param.beam_size]
        else:
            # N*beam x beam = N*beam x 1 + N*beam x beam
            accu_score = self.score.view(-1, 1) + topk_score
            accu_score = accu_score.view(self.param.batch_size, -1)
            # N x beam*beam => N x beam
            self.score, topk_index = th.topk(accu_score,
                                             self.param.beam_size,
                                             dim=-1)
            # point to father's node
            # N x beam
            self.point.append(topk_index // self.param.beam_size)
            # N x beam*beam
            topk_token = topk_token.view(self.param.batch_size, -1)
            token = th.gather(topk_token, -1, topk_index)
        self.token.append(token.clone())

    def _trace_back_hypos(self,
                          batch: int,
                          point: th.Tensor,
                          final: bool = False) -> List[Dict]:
        """
        Trace back the decoding transcription sequence from the current time point
        Args:
            batch (int): batch index
            point (Tensor): initial backward point
        """
        score = self.score[batch, point].tolist()
        self.score[batch, point] = NEG_INF
        points = [p[batch] for p in self.point]
        tokens = [t[batch] for t in self.token]
        return self.trace_hypos(score, points, tokens, final=final)
