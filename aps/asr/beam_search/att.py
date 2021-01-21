#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for attention based AM (RNN decoder)
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, List, Dict
from aps.asr.beam_search.utils import BeamSearchParam, BeamTracker, BatchBeamTracker
from aps.asr.beam_search.lm import lm_score_impl, adjust_hidden, LmType
from aps.utils import get_logger

logger = get_logger(__name__)


def greedy_search(decoder: nn.Module,
                  att_net: nn.Module,
                  enc_out: th.Tensor,
                  sos: int = -1,
                  eos: int = -1,
                  len_norm: bool = True) -> List[Dict]:
    """
    Greedy search (for debugging, should equal to beam search with #beam-size == 1)
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    N, _, D_enc = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    device = enc_out.device
    proj = th.zeros([N, D_enc], device=device)
    att_ali = None  # attention alignments
    dec_hid = None
    # zero init context
    att_ctx = th.zeros([N, D_enc], device=device)
    dec_tok = [sos]
    score = 0
    # clear states
    att_net.clear()
    while True:
        pre_tok = th.tensor([dec_tok[-1]], device=device)
        # make one step
        dec_out, att_ctx, dec_hid, att_ali, proj = decoder.step(att_net,
                                                                pre_tok,
                                                                enc_out,
                                                                att_ctx,
                                                                dec_hid=dec_hid,
                                                                att_ali=att_ali,
                                                                enc_len=None,
                                                                proj=proj)
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
                att_net: nn.Module,
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
                eos_threshold: float = 1) -> List[Dict]:
    """
    Vectorized beam search algothrim (see batch version beam_search_batch)
    Args
        att_net (nn.Module): attention network
        enc_out (Tensor): 1 x T x F, encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    N, T, D_enc = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")

    min_len = max(min_len, int(min_len_ratio * T))
    max_len = min(max_len, int(max_len_ratio * T))
    logger.info(f"--- shape of the encoder output: {T} x {D_enc}")
    logger.info("--- length constraint of the decoding " +
                f"sequence: ({min_len}, {max_len})")
    nbest = min(beam_size, nbest)
    device = enc_out.device
    att_ali = None
    dec_hid = None
    # N x T x F => N*beam x T x F
    enc_out = th.repeat_interleave(enc_out, beam_size, 0)
    att_ctx = th.zeros([N * beam_size, D_enc], device=device)
    proj = th.zeros([N * beam_size, D_enc], device=device)

    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 device=device,
                                 min_len=min_len,
                                 max_len=max_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 len_penalty=len_penalty,
                                 cov_penalty=cov_penalty,
                                 cov_threshold=cov_threshold,
                                 eos_threshold=eos_threshold)
    beam_tracker = BeamTracker(beam_param)

    lm_state = None
    # clear states
    att_net.clear()
    # step by step
    stop = False
    while not stop:
        # beam
        pre_tok, point = beam_tracker[-1]

        # step forward
        dec_hid = adjust_hidden(point, dec_hid)
        att_ali = None if att_ali is None else att_ali[point]
        dec_out, att_ctx, dec_hid, att_ali, proj = decoder.step(
            att_net,
            pre_tok,
            enc_out,
            att_ctx[point],
            dec_hid=dec_hid,
            att_ali=att_ali,
            proj=proj[point])
        # compute prob: beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)
        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0
        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob, att_ali=att_ali)
    # return nbest
    return beam_tracker.nbest_hypos(nbest)


def beam_search_batch(decoder: nn.Module,
                      att_net: nn.Module,
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
        att_net (nn.Module): attention network
        enc_out (Tensor): N x T x F, encoder output
        enc_len (Tensor): N, length of the encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam_size > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam_size}) > vocabulary size")

    N, T, D_enc = enc_out.shape
    min_len = [
        max(min_len, int(min_len_ratio * elen.item())) for elen in enc_len
    ]
    max_len = [
        min(max_len, int(max_len_ratio * elen.item())) for elen in enc_len
    ]
    logger.info(f"--- shape of the encoder output: {T} x {D_enc}")
    logger.info("--- length constraint of the decoding " +
                f"sequence: {[(i, j) for i, j in zip(min_len, max_len)]}")

    nbest = min(beam_size, nbest)
    device = enc_out.device
    att_ali = None
    dec_hid = None
    # N x T x F => N*beam x T x F
    enc_out = th.repeat_interleave(enc_out, beam_size, 0)
    enc_len = th.repeat_interleave(enc_len, beam_size, 0)
    att_ctx = th.zeros([N * beam_size, D_enc], device=device)
    proj = th.zeros([N * beam_size, D_enc], device=device)

    lm_state = None
    beam_param = BeamSearchParam(beam_size=beam_size,
                                 sos=sos,
                                 eos=eos,
                                 device=device,
                                 min_len=min_len,
                                 max_len=max_len,
                                 len_norm=len_norm,
                                 lm_weight=lm_weight,
                                 len_penalty=len_penalty,
                                 cov_penalty=cov_penalty,
                                 cov_threshold=cov_threshold,
                                 eos_threshold=eos_threshold)
    beam_tracker = BatchBeamTracker(N, beam_param)

    # clear states
    att_net.clear()
    # step by step
    stop = False
    while not stop:
        # N*beam
        pre_tok, point = beam_tracker[-1]
        # step forward
        dec_hid = adjust_hidden(point, dec_hid)
        att_ali = None if att_ali is None else att_ali[point]
        dec_out, att_ctx, dec_hid, att_ali, proj = decoder.step(
            att_net,
            pre_tok,
            enc_out,
            att_ctx[point],
            dec_hid=dec_hid,
            att_ali=att_ali,
            proj=proj[point])
        # compute prob: N*beam x V, nagetive
        am_prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm and beam_param.lm_weight > 0:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_tok, lm_state)
        else:
            lm_prob = 0

        # one beam search step
        stop = beam_tracker.step(am_prob, lm_prob, att_ali=att_ali)
    # return nbest
    return beam_tracker.nbest_hypos(nbest, auto_stop=stop)
