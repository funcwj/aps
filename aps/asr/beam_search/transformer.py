#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for transformer based AM (Transformer decoder)
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from torch.nn.utils.rnn import pad_sequence
from aps.asr.beam_search.utils import BeamSearchParam, BeamTracker, BatchBeamTracker
from aps.asr.beam_search.lm import lm_score_impl, LmType
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
    # T x N x D
    _, N, _ = enc_out.shape
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


def decoder_rescore(ctc_nbest: List[Dict],
                    decoder: nn.Module,
                    enc_out: th.Tensor,
                    ctc_weight: float = 0,
                    len_norm: bool = True) -> List[Dict]:
    """
    Rescore CTC nbest using transformer decoder
    Args:
        ctc_nbest: nbest from CTC beam search
        enc_out (Tensor): T x 1 x F
    """
    _, N, D_enc = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    device = enc_out.device
    nbest = len(ctc_nbest)
    logger.info(f"--- decoder rescoring for CTC {nbest}-best hypothesis, " +
                f"ctc_weight = {ctc_weight}")
    eos = ctc_nbest[0]["trans"][-1]
    # remove eos: [sos + ... + eos] => [sos + ...]
    ctc_seq = [th.as_tensor(h["trans"][:-1]) for h in ctc_nbest]
    tgt_pad = pad_sequence(ctc_seq, batch_first=True, padding_value=eos)
    # Ti x N x D => N x Ti x D
    enc_out = th.repeat_interleave(enc_out, nbest, 1).transpose(0, 1)
    # N x To x V
    dec_out = decoder(enc_out, None, tgt_pad.to(device), None)
    dec_score = th.log_softmax(dec_out, -1)
    # rescore
    rescore_nbest = []
    for i, hyp in enumerate(ctc_nbest):
        att_score = 0.0
        # e.g., <sos> a b c d <eos>, att_score adds for a b c d <eos>
        for n, w in enumerate(hyp["trans"][1:]):
            att_score += dec_score[i, n, w].item()
        fusion_score = hyp["score"] * ctc_weight + att_score
        rescore_nbest.append({
            "score": fusion_score / (len(hyp["trans"][1:]) if len_norm else 1),
            "trans": hyp["trans"]
        })
    return sorted(rescore_nbest, key=lambda n: n["score"], reverse=True)


def beam_search(decoder: nn.Module,
                enc_out: th.Tensor,
                lm: Optional[LmType] = None,
                ctc_prob: Optional[th.Tensor] = None,
                lm_weight: float = 0,
                beam_size: int = 8,
                nbest: int = 1,
                max_len: int = -1,
                max_len_ratio: float = 1,
                min_len: int = 0,
                min_len_ratio: float = 0,
                sos: int = -1,
                eos: int = -1,
                unk: int = -1,
                len_norm: bool = True,
                ctc_weight: float = 0,
                end_detect: bool = False,
                len_penalty: float = 0,
                cov_penalty: float = 0,
                temperature: float = 1,
                allow_partial: bool = False,
                cov_threshold: float = 0.5,
                eos_threshold: float = 0) -> List[Dict]:
    """
    Vectorized beam search algothrim for transformer decoder
    Args
        enc_out (Tensor): T x 1 x F, encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    T, N, D_enc = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")

    min_len = max(min_len, int(min_len_ratio * T))
    max_len = min(max_len, int(max_len_ratio * T)) if max_len_ratio > 0 else T
    logger.info(f"--- shape of the encoder output: {T} x {D_enc}")
    logger.info("--- length constraint of the decoding " +
                f"sequence: ({min_len}, {max_len})")
    nbest = min(beam_size, nbest)
    device = enc_out.device

    # cov_* are diabled
    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 unk=unk,
                                 device=device,
                                 min_len=min_len,
                                 max_len=max_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 ctc_weight=ctc_weight,
                                 end_detect=end_detect,
                                 len_penalty=len_penalty,
                                 allow_partial=allow_partial,
                                 eos_threshold=eos_threshold,
                                 ctc_beam_size=int(beam_size * 1.5))
    beam_tracker = BeamTracker(beam_param, ctc_prob=ctc_prob)
    pre_emb = None
    lm_state = None
    # T x 1 x D => T x beam x D
    enc_out = th.repeat_interleave(enc_out, beam_size, 1)
    # step by step
    stop = False
    while not stop:
        # beam
        pre_tok, point = beam_tracker[-1]
        # beam x V
        dec_out, pre_emb = decoder.step(
            enc_out,
            pre_tok[:, None],
            out_idx=-1,
            pre_emb=None if pre_emb is None else pre_emb[:, point])

        # compute prob: beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)
        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0
        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob)
    # return nbest
    return beam_tracker.nbest_hypos(nbest)


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
                      unk: int = -1,
                      len_norm: bool = True,
                      ctc_weight: float = 0,
                      end_detect: bool = False,
                      len_penalty: float = 0,
                      cov_penalty: float = 0,
                      temperature: float = 1,
                      allow_partial: bool = False,
                      cov_threshold: float = 0.5,
                      eos_threshold: float = 1) -> List[List[Dict]]:
    """
    Batch level vectorized beam search algothrim
    Args
        enc_out (Tensor): T x N x F, encoder output
        enc_len (Tensor): N, length of the encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")

    _, N, _ = enc_out.shape
    min_len = [
        max(min_len, int(min_len_ratio * elen.item())) for elen in enc_len
    ]
    max_len = [
        min(max_len, int(max_len_ratio *
                         elen.item())) if max_len_ratio > 0 else elen.item()
        for elen in enc_len
    ]
    logger.info("--- length constraint of the decoding " +
                f"sequence: {[(i, j) for i, j in zip(min_len, max_len)]}")

    nbest = min(beam_size, nbest)
    device = enc_out.device
    enc_len = th.repeat_interleave(enc_len, beam_size, 0)
    # T x N x D => T x N*beam x D
    enc_out = th.repeat_interleave(enc_out, beam_size, 1)

    pre_emb = None
    lm_state = None
    # cov_* are diabled
    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 unk=unk,
                                 device=device,
                                 min_len=min_len,
                                 max_len=max_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 ctc_weight=ctc_weight,
                                 end_detect=end_detect,
                                 len_penalty=len_penalty,
                                 allow_partial=allow_partial,
                                 eos_threshold=eos_threshold,
                                 ctc_beam_size=int(beam_size * 1.5))
    beam_tracker = BatchBeamTracker(N, beam_param)
    # step by step
    stop = False
    while not stop:
        # N*beam
        pre_tok, point = beam_tracker[-1]
        # beam x V
        dec_out, pre_emb = decoder.step(
            enc_out,
            pre_tok[:, None],
            out_idx=-1,
            pre_emb=None if pre_emb is None else pre_emb[:, point])
        # compute prob: N*beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0

        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob)
    # return nbest
    return beam_tracker.nbest_hypos(nbest, auto_stop=stop)
