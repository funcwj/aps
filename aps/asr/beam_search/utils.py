#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from typing import List, Dict, Union, Tuple, Optional
from aps.const import NEG_INF


class BaseBeamTracker(object):
    """
    Base class (to be inheried)
    """

    def __init__(self,
                 beam_size: int,
                 batch_size: Optional[int] = None,
                 sos: int = 1,
                 eos: int = 2,
                 device: Union[th.device, str] = "cpu",
                 penalty: float = 0,
                 normalized: bool = True) -> None:
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.sos = sos
        self.eos = eos
        self.penalty = penalty
        self.normalized = normalized


class BeamTracker(BaseBeamTracker):
    """
    A data structure used in beam search algothrim
    """

    def __init__(self,
                 beam_size: int,
                 sos: int = 1,
                 eos: int = 2,
                 device: Union[th.device, str] = "cpu",
                 penalty: float = 0,
                 normalized: bool = True) -> None:
        super(BeamTracker, self).__init__(beam_size,
                                          sos=sos,
                                          eos=eos,
                                          device=device,
                                          penalty=penalty,
                                          normalized=normalized)
        self.token = [th.tensor([sos] * beam_size, device=device)]
        self.point = [th.tensor(range(beam_size), device=device)]
        self.score = th.zeros(beam_size, device=device)

    def __getitem__(self, t: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Return the token and backward point
        """
        return (self.token[t], self.point[t])

    def trace_back(self, final: bool = False) -> Optional[List[Dict]]:
        """
        Return end flags
        """
        end_eos = (self.token[-1] == self.eos).tolist()
        hyp = None
        if not final and sum(end_eos):
            idx = [i for i, end_with_eos in enumerate(end_eos) if end_with_eos]
            idx = th.tensor(idx, device=self.score.device)
            hyp = self._trace_back_hypos(idx, final=False)
        not_end = [not f for f in end_eos]
        if final and sum(not_end):
            idx = th.tensor([i for i, go_on in enumerate(not_end) if go_on],
                            device=self.score.device)
            hyp = self._trace_back_hypos(idx, final=True)
        return hyp

    def prune_beam(self, log_prob: th.Tensor):
        """
        Prune and update score & token & backward point
        Args:
            log_prob (Tensor): N x V, log prob
        """
        # local pruning: beam x beam
        topk_score, topk_token = th.topk(log_prob, self.beam_size, dim=-1)
        if len(self.point) == 1:
            self.score += topk_score[0]
            self.token.append(topk_token[0])
            self.point.append(self.point[-1])
        else:
            # beam*beam
            topk_token = topk_token.view(-1)
            # beam*beam
            accu_score = (self.score[..., None] + topk_score).view(-1)
            # beam*beam => beam
            self.score, topk_index = th.topk(accu_score, self.beam_size, dim=-1)
            # point to father's node
            self.point.append(topk_index // self.beam_size)
            self.token.append(topk_token[topk_index])

    def _trace_back_hypos(self,
                          point: th.Tensor,
                          final: bool = False) -> List[Dict]:
        """
        Trace back the decoding transcription sequence from the current time point
        Args:
            point (Tensor): initial backward point
        """
        trans = []
        score = self.score[point].tolist()
        self.score[point] = NEG_INF
        for ptr, tok in zip(self.point[::-1], self.token[::-1]):
            trans.append(tok[point].tolist())
            point = ptr[point]
        hypos = []
        trans = trans[::-1]
        for i, s in enumerate(score):
            token = [t[i] for t in trans]
            token_len = len(token) if final else len(token) - 1
            score = s + token_len * self.penalty
            hypos.append({
                "score": score / (token_len if self.normalized else 1),
                "trans": token + [self.eos] if final else token
            })
        return hypos


class BatchBeamTracker(BaseBeamTracker):
    """
    A data structure used in batch version of the beam search
    """

    def __init__(self,
                 beam_size: int,
                 batch_size: int,
                 sos: int = 1,
                 eos: int = 2,
                 device: Union[th.device, str] = "cpu",
                 penalty: float = 0,
                 normalized: bool = True) -> None:
        super(BatchBeamTracker, self).__init__(beam_size,
                                               batch_size=batch_size,
                                               sos=sos,
                                               eos=eos,
                                               device=device,
                                               penalty=penalty,
                                               normalized=normalized)
        self.token = [
            th.tensor([[sos] * beam_size] * batch_size, device=device)
        ]
        self.point = [
            th.tensor([list(range(beam_size))] * batch_size, device=device)
        ]
        self.score = th.zeros(batch_size, beam_size, device=device)
        self.step_point = th.arange(0,
                                    beam_size * batch_size,
                                    beam_size,
                                    device=device)

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
        end_eos = (self.token[-1][batch] == self.eos).tolist()
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
        return hyp

    def prune_beam(self, log_prob: th.Tensor):
        """
        Prune and update score & token & backward point
        Args:
            log_prob (Tensor): N x V, log prob
        """
        # local pruning: N*beam x beam
        topk_score, topk_token = th.topk(log_prob, self.beam_size, dim=-1)
        if len(self.point) == 1:
            # N x beam
            self.score += topk_score[::self.beam_size]
            self.point.append(self.point[-1])
            token = topk_token[::self.beam_size]
        else:
            # N*beam x beam = N*beam x 1 + N*beam x beam
            accu_score = self.score.view(-1, 1) + topk_score
            accu_score = accu_score.view(self.batch_size, -1)
            # N x beam*beam => N x beam
            self.score, topk_index = th.topk(accu_score, self.beam_size, dim=-1)
            # point to father's node
            # N x beam
            self.point.append(topk_index // self.beam_size)
            # N x beam*beam
            topk_token = topk_token.view(self.batch_size, -1)
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
        trans = []
        score = self.score[batch, point].tolist()
        self.score[batch, point] = NEG_INF

        points = [p[batch] for p in self.point]
        tokens = [t[batch] for t in self.token]

        for ptr, tok in zip(points[::-1], tokens[::-1]):
            trans.append(tok[point].tolist())
            point = ptr[point]
        hypos = []
        trans = trans[::-1]
        for i, s in enumerate(score):
            token = [t[i] for t in trans]
            token_len = len(token) if final else len(token) - 1
            score = s + token_len * self.penalty
            hypos.append({
                "score": score / (token_len if self.normalized else 1),
                "trans": token + [self.eos] if final else token
            })
        return hypos
