#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional, NoReturn
from aps.const import NEG_INF
from aps.utils import get_logger

logger = get_logger(__name__)
verbose = False
double_check = False


@dataclass
class BeamSearchParam(object):
    """
    Parameters used in beam search
    """
    beam_size: int = 8
    sos: int = 1
    eos: int = 2
    # for batch version, it's list
    min_len: Union[int, List[int]] = 1
    max_len: Union[int, List[int]] = 1000
    lm_weight: float = 0
    eos_threshold: float = 0
    device: Union[th.device, str] = "cpu"
    len_penalty: float = 0
    cov_method: str = "v1"
    cov_penalty: float = 0
    cov_threshold: float = 0.5
    len_norm: bool = True
    end_detect: bool = False


class BaseBeamTracker(object):
    """
    Base class for implementation of the beam search (to be inheried)
    """

    def __init__(self, param: BeamSearchParam) -> None:
        self.param = param
        self.align = None  # B x T x U
        self.trans = None  # B x U
        self.none_eos_idx = None
        self.step_num = 0

    def end_detect(self,
                   hyp_ended: List[Dict],
                   look_back: int = 3,
                   end_threshold: float = 10) -> bool:
        """
        To stop beam search before reaching max_len
        """
        if not len(hyp_ended):
            return False
        global_best = max([h["score"] for h in hyp_ended])
        count = 0
        for l in range(self.step_num - look_back, self.step_num):
            hset = [h for h in hyp_ended if len(h["trans"]) - 1 == l]
            if len(hset):
                cur_best = max([h["score"] for h in hset])
                if global_best - cur_best >= end_threshold:
                    count += 1
        return count == look_back

    def concat(self, prev_: Optional[th.Tensor], point: Optional[th.Tensor],
               next_: Optional[th.Tensor]) -> th.Tensor:
        """
        Concat the alignment or transcription step by step
        Args:
            prev_ (Tensor): N x ... x U
            next_ (Tensor): N x ...
            point (Tensor): traceback point
        """
        if next_ is None:
            return prev_
        elif prev_ is None:
            return next_[..., None]
        elif point is None:
            return th.cat([prev_, next_[..., None]], -1)
        else:
            return th.cat([prev_[point], next_[..., None]], -1)

    def coverage(self, att_ali: Optional[th.Tensor]) -> Union[th.Tensor, float]:
        """
        Compute coverage score (found 2 way for computation)
        Args:
            att_ali (Tensor): N x T, alignment score (weight)
        Return
            cov_score: coverage score
        """
        if att_ali is None or self.param.cov_penalty <= 0:
            cov_score = 0
        else:
            assert att_ali is not None
            # sum over V, N x T
            att_sum_vocab = th.sum(self.align, -1)
            # N x 1
            if self.param.cov_method == "v2":
                cov = th.clamp_max(att_sum_vocab,
                                   self.param.cov_threshold).log()
            else:
                cov = (att_sum_vocab > self.param.cov_threshold).float()
            # sum over T
            cov_score = th.sum(cov, -1, keepdim=True)
        return cov_score * self.param.cov_penalty

    def beam_select(self, am_prob: th.Tensor,
                    lm_prob: Union[th.Tensor, float]) -> Tuple[th.Tensor]:
        """
        Perform beam selection
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
        Return:
            topk_score (Tensor): N x K, topk score
            topk_token (Tensor): N x K, topk token ID
        """
        # N x V, AM + LM
        fusion_prob = am_prob + self.param.lm_weight * lm_prob
        # process eos
        if self.param.eos_threshold > 0:
            if self.none_eos_idx is None:
                none_eos_idx = [
                    i for i in range(fusion_prob.shape[-1])
                    if i != self.param.eos
                ]
                self.none_eos_idx = th.tensor(none_eos_idx,
                                              device=am_prob.device)
            # current eos score
            eos_prob = fusion_prob[:, self.param.eos]
            # none_eos best score
            none_eos_best, _ = th.max(fusion_prob[:, self.none_eos_idx], dim=-1)
            # set inf to disable the eos
            disable_eos = eos_prob < none_eos_best * self.param.eos_threshold
            fusion_prob[disable_eos, self.param.eos] = NEG_INF
            if verbose and th.sum(disable_eos):
                disable_index = [i for i, s in enumerate(disable_eos) if s]
                logger.info(f"--- disable <eos> in beam index: {disable_index}")
        # local pruning: N*beam x beam
        topk_score, topk_token = th.topk(fusion_prob,
                                         self.param.beam_size,
                                         dim=-1)
        return (topk_score, topk_token)

    def trace_hypos(self,
                    point: th.Tensor,
                    score: List[float],
                    trans: th.Tensor,
                    align: Optional[th.Tensor],
                    point_list: List[th.Tensor],
                    token_list: List[th.Tensor],
                    final: bool = False) -> List[Dict]:
        """
        Traceback decoding hypothesis
        Args:
            point (Tensor): starting traceback point
            score (list[float]): final decoding score
            trans (Tensor): traced transcriptions (another way)
            align (Tensor): traced alignments
            point_list (list[Tensor]): traceback point
            token_list (list[Tensor]): token sequence
            final (bool): is final step or not
        """
        if align is not None:
            align = align[point].cpu()
        final_trans = trans[point].tolist()
        # we compute final sequence in two ways to verify the implmentation of beam search
        if double_check:
            check_trans = []
            for ptr, tok in zip(point_list[::-1], token_list[::-1]):
                check_trans.append(tok[point].tolist())
                point = ptr[point]
            check_trans = check_trans[::-1]
        hypos = []
        for i, s in enumerate(score):
            token = final_trans[i]
            # NOTE: double check implementation of the beam search
            assert not double_check or token == [t[i] for t in check_trans]
            token_len = len(token) if final else len(token) - 1
            score = s + token_len * self.param.len_penalty
            hypos.append({
                "score": score / (token_len if self.param.len_norm else 1),
                "trans": token + [self.param.eos] if final else token,
                "align": None if align is None else align[i]
            })
        return hypos

    def step(self,
             am_prob: th.Tensor,
             lm_prob: Union[th.Tensor, float],
             att_ali: Optional[th.Tensor] = None) -> bool:
        """
        Make one beam search step
        """
        raise NotImplementedError

    def nbest_hypos(self, nbest: int) -> List[Dict]:
        """
        Return nbest sequence
        """
        raise NotImplementedError


class BeamTracker(BaseBeamTracker):
    """
    A data structure used in beam search algothrim
    """

    def __init__(self, param: BeamSearchParam) -> None:
        super(BeamTracker, self).__init__(param)
        init_sos = th.tensor([param.sos] * param.beam_size, device=param.device)
        # decoded sequence upto current step
        self.trans = init_sos[:, None]
        # decoded token at each time step (unordered)
        self.token = [init_sos]
        # traceback point at each time step
        self.point = [th.tensor(range(param.beam_size), device=param.device)]
        # decoding score (upto current step)
        self.score = th.zeros(param.beam_size, device=param.device)
        self.hypos = []
        self.auto_stop = False
        self.acmu_score = th.zeros_like(self.score)

    def __getitem__(self, t: int) -> Tuple[th.Tensor, th.Tensor]:
        """
        Return the token and backward point
        """
        return (self.token[t], self.point[t])

    def _trace_back(self, final: bool = False) -> Optional[List[Dict]]:
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
        if hyp:
            hyp = [h for h in hyp if len(h["trans"]) >= self.param.min_len + 2]
            if verbose:
                for h in hyp:
                    logger.info("--- beam search gets decoding sequence " +
                                f"{h['trans']}, score = {h['score']:.2f}")
        return hyp

    def _init_search(self,
                     am_prob: th.Tensor,
                     lm_prob: Union[th.Tensor, float],
                     att_ali: Optional[th.Tensor] = None) -> NoReturn:
        """
        Kick off the beam search (to be used at the first step)
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        """
        assert len(self.point) == 1 and self.step_num == 0
        # local pruning: beam x V => beam x beam
        topk_score, topk_token = self.beam_select(am_prob, lm_prob)
        self.score += topk_score[0]
        self.acmu_score += topk_score[0]
        self.token.append(topk_token[0])
        self.point.append(self.point[-1])
        self.trans = self.concat(self.trans, None, topk_token[0])
        self.align = None if att_ali is None else att_ali[..., None]

    def _step_search(self,
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
        topk_score, topk_token = self.beam_select(am_prob, lm_prob)
        # beam x beam
        acmu_score = self.acmu_score[..., None] + topk_score
        score = acmu_score + self.coverage(att_ali)
        # beam*beam => beam
        self.score, topk_index = th.topk(score.view(-1),
                                         self.param.beam_size,
                                         dim=-1)
        # update accumulated score (AM + LM)
        self.acmu_score = acmu_score.view(-1)[topk_index]
        # point to father's node
        token = topk_token.view(-1)[topk_index]
        point = topk_index // self.param.beam_size
        # concat stats
        self.trans = self.concat(self.trans, point, token)
        self.align = self.concat(self.align, None,
                                 None if att_ali is None else att_ali[point])
        # collect
        self.token.append(token)
        self.point.append(point)

    def _trace_back_hypos(self,
                          point: th.Tensor,
                          final: bool = False) -> List[Dict]:
        """
        Trace back the decoding transcription sequence from the current time point
        Args:
            point (Tensor): initial backward point
        """
        score = self.score[point].tolist()
        self.acmu_score[point] = NEG_INF
        return self.trace_hypos(point,
                                score,
                                self.trans,
                                self.align,
                                self.point,
                                self.token,
                                final=final)

    def step(self,
             am_prob: th.Tensor,
             lm_prob: Union[th.Tensor, float],
             att_ali: Optional[th.Tensor] = None) -> bool:
        """
        Run one beam search step
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        Return:
            stop (bool): stop beam search or not
        """
        # local pruning
        if self.step_num == 0:
            self._init_search(am_prob, lm_prob, att_ali=att_ali)
        else:
            self._step_search(am_prob, lm_prob, att_ali=att_ali)
        self.step_num += 1
        # trace back ended sequence (process eos nodes)
        hyp_ended = self._trace_back(final=False)
        stop = False
        if hyp_ended:
            self.hypos += hyp_ended
            # all eos, stop beam search
            if len(hyp_ended) == self.param.beam_size:
                logger.info(
                    f"--- beam search ends (all eos) at step {self.step_num}")
                stop = True
            # auto detected
            if self.param.end_detect and self.end_detect(self.hypos):
                logger.info(
                    f"--- beam search ends (detected) at step {self.step_num}")
                stop = True
        # update auto_step flag
        self.auto_stop = stop
        # if reach max_len, also return true
        return stop or self.step_num == self.param.max_len

    def nbest_hypos(self, nbest: int) -> List[Dict]:
        """
        Return nbest sequence
        Args:
            nbest (int): nbest size
        """
        # not auto stop, add unfinished hypos
        if not self.auto_stop:
            logger.info("--- beam search reaches the final step ...")
            hyp_final = self._trace_back(final=True)
            if hyp_final:
                self.hypos += hyp_final
        # sort and get nbest
        logger.info(f"--- beam search gets top-{nbest} list " +
                    f"from {len(self.hypos)} hypos ...")
        sort_hypos = sorted(self.hypos, key=lambda n: n["score"], reverse=True)
        return sort_hypos[:nbest]


class BatchBeamTracker(BaseBeamTracker):
    """
    A data structure used in batch version of the beam search
    """

    def __init__(self, batch_size: int, param: BeamSearchParam) -> None:
        super(BatchBeamTracker, self).__init__(param)
        self.param = param
        self.batch_size = batch_size
        init_sos = th.tensor([param.sos] * param.beam_size * batch_size,
                             device=param.device)
        # [batch*beam, T]
        self.trans = init_sos[:, None]
        # [batch x beam, ...]
        self.token = [init_sos.view(batch_size, -1)]
        self.point = [
            th.tensor([list(range(param.beam_size))] * batch_size,
                      device=param.device)
        ]
        self.score = th.zeros(batch_size, param.beam_size, device=param.device)
        self.hypos = [[] for _ in range(batch_size)]
        self.auto_stop = [False] * batch_size
        self.acmu_score = th.zeros_like(self.score)
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

    def _trace_back(self, batch, final: bool = False) -> Optional[List[Dict]]:
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
        if hyp:
            hyp = [
                h for h in hyp
                if len(h["trans"]) >= self.param.min_len[batch] + 2
            ]
            if verbose:
                for h in hyp:
                    logger.info(
                        "--- beam search gets decoding sequence " +
                        f"(batch[{batch}]) {h['trans']}, score = {h['score']:.2f}"
                    )
        return hyp

    def _init_search(self,
                     am_prob: th.Tensor,
                     lm_prob: Union[th.Tensor, float],
                     att_ali: Optional[th.Tensor] = None) -> NoReturn:
        """
        Kick off the beam search (to be used at the first step)
        Args:
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        """
        assert len(self.point) == 1 and self.step_num == 0
        # local pruning: N*beam x V => N*beam x beam
        topk_score, topk_token = self.beam_select(am_prob, lm_prob)
        init_score = topk_score[::self.param.beam_size]
        init_token = topk_token[::self.param.beam_size]
        # N x beam
        self.score += init_score
        self.acmu_score += init_score
        self.token.append(init_token.clone())
        self.point.append(self.point[-1])
        self.trans = self.concat(self.trans, None, self.token[-1].view(-1))
        self.align = None if att_ali is None else att_ali[..., None]

    def _step_search(self,
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
        topk_score, topk_token = self.beam_select(am_prob, lm_prob)
        # N*beam x beam = N*beam x 1 + N*beam x beam
        acmu_score = self.acmu_score.view(-1, 1) + topk_score
        score = acmu_score + self.coverage(att_ali)
        # N x beam*beam => N x beam
        self.score, topk_index = th.topk(score.view(self.batch_size, -1),
                                         self.param.beam_size,
                                         dim=-1)
        # update accmulated score (AM + LM)
        self.acmu_score = th.gather(acmu_score.view(self.batch_size, -1), -1,
                                    topk_index)
        # N x beam, point to father's node
        point = topk_index // self.param.beam_size
        # N x beam*beam => N x beam
        token = th.gather(topk_token.view(self.batch_size, -1), -1, topk_index)
        # append stats
        self.token.append(token.clone())
        self.point.append(point)
        token, point = self[-1]
        # concat stats
        self.trans = self.concat(self.trans, point, token)
        self.align = self.concat(self.align, None,
                                 None if att_ali is None else att_ali[point])

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
        self.acmu_score[batch, point] = NEG_INF
        trans = th.chunk(self.trans, self.batch_size, 0)[batch]
        align = th.chunk(self.align, self.batch_size, 0)[batch]
        points = [p[batch] for p in self.point]
        tokens = [t[batch] for t in self.token]
        return self.trace_hypos(point,
                                score,
                                trans,
                                align,
                                points,
                                tokens,
                                final=final)

    def step(self,
             am_prob: th.Tensor,
             lm_prob: Union[th.Tensor, float],
             att_ali: Optional[th.Tensor] = None) -> bool:
        """
        Run one beam search step
        Args:
            step_num (int): step number
            am_prob (Tensor): N x V, acoustic prob
            lm_prob (Tensor): N x V, language prob
            att_ali (Tensor): N x T, alignment score (weight)
        Return:
            stop (bool): stop beam search or not
        """
        # local pruning
        if self.step_num == 0:
            self._init_search(am_prob, lm_prob, att_ali=att_ali)
        else:
            self._step_search(am_prob, lm_prob, att_ali=att_ali)
        self.step_num += 1
        # trace back ended sequence (process eos nodes)
        for u in range(self.batch_size):
            # reach the max_len of utterance u, skip
            max_len = self.param.max_len[u]
            if self.step_num >= max_len:
                if self.step_num == max_len:
                    logger.info(f"--- beam search (batch[{u}]) reaches " +
                                f"max_len {max_len}")
                continue
            hyp_ended = self._trace_back(u, final=False)
            if hyp_ended:
                self.hypos[u] += hyp_ended
                # all eos, stop beam search
                if len(hyp_ended) == self.param.beam_size:
                    logger.info(
                        f"--- beam search ends (all eos, batch[{u}]) at " +
                        f"step {self.step_num}")
                    self.auto_stop[u] = True
                # auto detected
                if self.param.end_detect and self.end_detect(self.hypos[u]):
                    logger.info(
                        f"--- beam search ends (detected, batch[{u}]) at " +
                        f"step {self.step_num}")
                    self.auto_stop[u] = True
        # all True, stop search
        stop = sum(self.auto_stop) == self.batch_size
        if stop:
            logger.info(
                f"--- beam search (all batches) ends at step {self.step_num}")
        # if reach max(max_len), also return true to stop batch beam search
        return stop or self.step_num == max(self.param.max_len)

    def nbest_hypos(self,
                    nbest: int,
                    auto_stop: bool = True) -> List[List[Dict]]:
        """
        Return nbest sequence
        Args:
            nbest (int): nbest size
            auto_stop: beam search is auto-stopped or not
        """
        # not auto stop, add unfinished hypos
        if not auto_stop:
            for u in range(self.batch_size):
                # skip utterance u
                if self.auto_stop[u]:
                    continue
                # process end
                logger.info("--- beam search reaches the final step " +
                            f"for batch[{u}] ...")
                hyp_final = self._trace_back(u, final=True)
                if hyp_final:
                    self.hypos[u] += hyp_final
        # sort and get nbest
        nbest_batch = []
        for u, utt_bypos in enumerate(self.hypos):
            logger.info(f"--- beam search gets top-{nbest} list (batch[{u}]) " +
                        f"from {len(utt_bypos)} hypos ...")
            sort_hypos = sorted(utt_bypos,
                                key=lambda n: n["score"],
                                reverse=True)
            nbest_batch.append(sort_hypos[:nbest])
        return nbest_batch
