#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for transformer based AM (Transformer decoder)
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.beam_search.utils import BeamSearchParam, BeamTracker, BatchBeamTracker
from aps.asr.beam_search.lm import rnnlm_score, ngram_score, LmType

from typing import List, Dict, Optional


def beam_search(decoder: nn.Module,
                enc_out: th.Tensor,
                lm: Optional[LmType] = None,
                lm_weight: float = 0,
                beam_size: int = 8,
                nbest: int = 1,
                max_len: int = -1,
                min_len: int = 0,
                sos: int = -1,
                eos: int = -1,
                len_norm: bool = True,
                len_penalty: float = 0,
                cov_penalty: float = 0,
                temperature: float = 1,
                cov_threshold: float = 0.5,
                eos_threshold: float = 0) -> List[Dict]:
    """
    Vectorized beam search algothrim for transformer decoder
    Args
        enc_out (Tensor): 1 x T x F, encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if max_len <= 0:
        raise RuntimeError(f"Invalid max_len: {max_len:d}")
    N, _, _ = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")

    if lm:
        if isinstance(lm, nn.Module):
            lm_score_impl = rnnlm_score
        else:
            lm_score_impl = ngram_score

    nbest = min(beam_size, nbest)
    device = enc_out.device

    # cov_* are diabled
    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 device=device,
                                 min_len=min_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 len_penalty=len_penalty,
                                 eos_threshold=eos_threshold)
    beam_tracker = BeamTracker(beam_param)
    hypos = []
    pre_emb = None
    lm_state = None
    # Ti x beam x D
    enc_out = th.repeat_interleave(enc_out, beam_size, 1)
    # step by step
    for t in range(max_len):
        # beam
        pre_out, point = beam_tracker[-1] if t else beam_tracker[0]
        # beam x V
        dec_out, pre_emb = decoder.step(enc_out,
                                        pre_out[:, None],
                                        out_idx=-1,
                                        pre_emb=pre_emb,
                                        point=point)

        # compute prob: beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_out, lm_state)
        else:
            lm_prob = 0

        # finished sequence
        hyp_ended = beam_tracker.step(t, am_prob, lm_prob)

        # process eos nodes
        if hyp_ended:
            hypos += hyp_ended

        if len(hypos) >= beam_size:
            break

        # process non-eos nodes at the final step
        if t == max_len - 1:
            hyp_final = beam_tracker.trace_back(final=True)
            if hyp_final:
                hypos += hyp_final

    nbest_hypos = sorted(hypos, key=lambda n: n["score"], reverse=True)
    return nbest_hypos[:nbest]


def beam_search_batch(decoder: nn.Module,
                      enc_out: th.Tensor,
                      enc_len: th.Tensor,
                      lm: Optional[LmType] = None,
                      lm_weight: float = 0,
                      beam_size: int = 8,
                      nbest: int = 1,
                      max_len: int = -1,
                      min_len: int = 0,
                      sos: int = -1,
                      eos: int = -1,
                      len_norm: bool = True,
                      len_penalty: float = 0,
                      cov_penalty: float = 0,
                      temperature: float = 1,
                      cov_threshold: float = 0.5,
                      eos_threshold: float = 1) -> List[Dict]:
    """
    Batch level vectorized beam search algothrim
    Args
        enc_out (Tensor): N x T x F, encoder output
        enc_len (Tensor): N, length of the encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if max_len <= 0:
        raise RuntimeError(f"Invalid max_len: {max_len:d}")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")
    if lm:
        if isinstance(lm, nn.Module):
            lm_score_impl = rnnlm_score
        else:
            lm_score_impl = ngram_score

    N, _, _ = enc_out.shape
    nbest = min(beam_size, nbest)
    device = enc_out.device
    # N x T x F => N*beam x T x F
    enc_out = th.repeat_interleave(enc_out, beam_size, 1)
    enc_len = th.repeat_interleave(enc_len, beam_size, 0)

    pre_emb = None
    lm_state = None
    # cov_* are diabled
    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 device=device,
                                 min_len=min_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 len_penalty=len_penalty,
                                 eos_threshold=eos_threshold)
    beam_tracker = BatchBeamTracker(N, beam_param)
    # for each utterance
    hypos = [[] for _ in range(N)]
    stop_batch = [False] * N
    # clear states
    # step by step
    for t in range(max_len):
        # N*beam
        pre_out, point = beam_tracker[-1] if t else beam_tracker[0]
        # beam x V
        dec_out, pre_emb = decoder.step(enc_out,
                                        pre_out[:, None],
                                        out_idx=-1,
                                        pre_emb=pre_emb,
                                        point=point)
        # compute prob: N*beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_out, lm_state)
        else:
            lm_prob = 0

        # local pruning: N*beam x beam
        if t == 0:
            beam_tracker.init_beam(am_prob, lm_prob)
        else:
            beam_tracker.prune_beam(am_prob, lm_prob)

        # process eos nodes
        for u in range(N):
            hyp_ended = beam_tracker.trace_back(u, final=False)
            if hyp_ended:
                hypos[u] += hyp_ended

            if len(hypos[u]) >= beam_size:
                stop_batch[u] = True

        # all True, break search
        if sum(stop_batch) == N:
            break

        # process non-eos nodes at the final step
        if t == max_len - 1:
            for u in range(N):
                # skip utterance u
                if stop_batch[u]:
                    continue
                # process end
                hyp_final = beam_tracker.trace_back(u, final=True)
                if hyp_final:
                    hypos[u] += hyp_final

    nbest_hypos = []
    for utt_bypos in hypos:
        hypos = sorted(utt_bypos, key=lambda n: n["score"], reverse=True)
        nbest_hypos.append(hypos[:nbest])
    return nbest_hypos
