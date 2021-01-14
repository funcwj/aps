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
from aps.utils import get_logger
from typing import List, Dict, Optional

logger = get_logger(__name__)


def greedy_search(decoder: nn.Module,
                  enc_out: th.Tensor,
                  sos: int = -1,
                  eos: int = -1,
                  len_norm: bool = True) -> List[Dict]:
    """
    Greedy search (for debugging, should equal to beam search with #beam-size == 1)
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    N, _, _ = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    device = enc_out.device
    dec_tok = [sos]
    pre_emb = None
    score = 0
    while True:
        pre_tok = th.tensor([dec_tok[-1]], device=device)
        # make one step
        dec_out, pre_emb = decoder.step(enc_out,
                                        pre_tok[:, None],
                                        out_idx=-1,
                                        pre_emb=pre_emb)
        prob = tf.log_softmax(dec_out, dim=-1)
        pred_score, pred_token = th.topk(prob, 1, dim=-1)
        dec_tok.append(pred_token.item())
        score += pred_score.item()
        if dec_tok[-1] == eos:
            break
    return [{
        "score": score / (len(dec_tok) - 1) if len_norm else score,
        "trans": dec_tok
    }]


def beam_search(decoder: nn.Module,
                enc_out: th.Tensor,
                lm: Optional[LmType] = None,
                lm_weight: float = 0,
                beam_size: int = 8,
                nbest: int = 1,
                max_len: int = -1,
                max_len_ratio: float = 1,
                min_len: int = 0,
                min_len_ratio: float = 0,
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
    N, T, _ = enc_out.shape
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

    min_len = max(min_len, int(min_len_ratio * T))
    max_len = min(max_len, int(max_len_ratio * T))
    logger.info("--- length constraint of the decoding " +
                f"sequence: ({min_len}, {max_len})")
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
    pre_emb = None
    lm_state = None
    # Ti x beam x D
    enc_out = th.repeat_interleave(enc_out, beam_size, 1)
    # step by step
    stop = False
    for _ in range(max_len):
        # beam
        pre_tok, point = beam_tracker[-1]
        # beam x V
        dec_out, pre_emb = decoder.step(enc_out,
                                        pre_tok[:, None],
                                        out_idx=-1,
                                        pre_emb=pre_emb,
                                        point=point)

        # compute prob: beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0

        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob)
        if stop:
            break

    return beam_tracker.nbest_hypos(nbest, auto_stop=stop)


def beam_search_batch(decoder: nn.Module,
                      enc_out: th.Tensor,
                      enc_len: th.Tensor,
                      lm: Optional[LmType] = None,
                      lm_weight: float = 0,
                      beam_size: int = 8,
                      nbest: int = 1,
                      max_len: int = -1,
                      max_len_ratio: float = 1,
                      min_len: int = 0,
                      min_len_ratio: float = 0,
                      sos: int = -1,
                      eos: int = -1,
                      len_norm: bool = True,
                      len_penalty: float = 0,
                      cov_penalty: float = 0,
                      temperature: float = 1,
                      cov_threshold: float = 0.5,
                      eos_threshold: float = 1) -> List[List[Dict]]:
    """
    Batch level vectorized beam search algothrim
    Args
        enc_out (Tensor): N x T x F, encoder output
        enc_len (Tensor): N, length of the encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")
    if lm:
        if isinstance(lm, nn.Module):
            lm_score_impl = rnnlm_score
        else:
            lm_score_impl = ngram_score

    N, T, _ = enc_out.shape
    min_len = max(min_len, int(min_len_ratio * T))
    max_len = min(max_len, int(max_len_ratio * T))
    logger.info("--- length constraint of the decoding " +
                f"sequence: ({min_len}, {max_len})")
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
    # step by step
    stop = False
    for _ in range(max_len):
        # N*beam
        pre_tok, point = beam_tracker[-1]
        # beam x V
        dec_out, pre_emb = decoder.step(enc_out,
                                        pre_tok[:, None],
                                        out_idx=-1,
                                        pre_emb=pre_emb,
                                        point=point)
        # compute prob: N*beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0

        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob)
        if stop:
            break

    return beam_tracker.nbest_hypos(nbest, auto_stop=stop)
