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
from aps.asr.beam_search.lm import rnnlm_score, ngram_score, LmType


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
        pred, att_ctx, dec_hid, att_ali, proj = decoder.step(att_net,
                                                             pre_tok,
                                                             enc_out,
                                                             att_ctx,
                                                             dec_hid=dec_hid,
                                                             att_ali=att_ali,
                                                             enc_len=None,
                                                             proj=proj)
        prob = tf.log_softmax(pred, dim=-1)
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
                beam: int = 8,
                nbest: int = 1,
                max_len: int = -1,
                sos: int = -1,
                eos: int = -1,
                penalty: float = 0,
                coverage: float = 0,
                len_norm: bool = True,
                temperature: float = 1) -> List[Dict]:
    """
    Vectorized beam search algothrim (see batch version beam_search_batch)
    Args
        att_net (nn.Module): attention network
        enc_out (Tensor): 1 x T x F, encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if max_len <= 0:
        raise RuntimeError(f"Invalid max_len: {max_len:d}")
    N, _, D_enc = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam}) > vocabulary size")

    if lm:
        if isinstance(lm, nn.Module):
            lm_score_impl = rnnlm_score
        else:
            lm_score_impl = ngram_score

    nbest = min(beam, nbest)
    device = enc_out.device
    att_ali = None
    dec_hid = None
    # N x T x F => N*beam x T x F
    enc_out = th.repeat_interleave(enc_out, beam, 0)
    att_ctx = th.zeros([N * beam, D_enc], device=device)
    proj = th.zeros([N * beam, D_enc], device=device)

    beam_tracker = BeamTracker(
        BeamSearchParam(beam_size=beam,
                        sos=sos,
                        eos=eos,
                        device=device,
                        penalty=penalty,
                        coverage=coverage,
                        lm_weight=lm_weight,
                        len_norm=len_norm))

    lm_state = None
    hypos = []
    # clear states
    att_net.clear()
    # step by step
    for t in range(max_len):
        # beam
        pre_out, point = beam_tracker[-1] if t else beam_tracker[0]
        # step forward
        pred, att_ctx, dec_hid, att_ali, proj = decoder.step(att_net,
                                                             pre_out,
                                                             enc_out,
                                                             att_ctx,
                                                             dec_hid=dec_hid,
                                                             att_ali=att_ali,
                                                             proj=proj,
                                                             point=point)
        # compute prob: beam x V, nagetive
        am_prob = tf.log_softmax(pred / temperature, dim=-1)

        if lm:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_out, lm_state)
        else:
            lm_prob = 0

        # local pruning
        beam_tracker.prune_beam(am_prob, lm_prob, att_ali=att_ali)
        # continue flags
        hyp_ended = beam_tracker.trace_back(final=False)

        # process eos nodes
        if hyp_ended:
            hypos += hyp_ended

        if len(hypos) >= beam:
            break

        # process non-eos nodes at the final step
        if t == max_len - 1:
            hyp_final = beam_tracker.trace_back(final=True)
            if hyp_final:
                hypos += hyp_final

    nbest_hypos = sorted(hypos, key=lambda n: n["score"], reverse=True)
    return nbest_hypos[:nbest]


def beam_search_batch(decoder: nn.Module,
                      att_net: nn.Module,
                      enc_out: th.Tensor,
                      enc_len: th.Tensor,
                      lm: Optional[LmType] = None,
                      lm_weight: float = 0,
                      beam: int = 8,
                      nbest: int = 1,
                      max_len: int = -1,
                      sos: int = -1,
                      eos: int = -1,
                      coverage: float = 0,
                      len_norm: bool = True,
                      penalty: float = 0,
                      temperature: float = 1) -> List[Dict]:
    """
    Batch level vectorized beam search algothrim
    Args
        att_net (nn.Module): attention network
        enc_out (Tensor): N x T x F, encoder output
        enc_len (Tensor): N, length of the encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if max_len <= 0:
        raise RuntimeError(f"Invalid max_len: {max_len:d}")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam}) > vocabulary size")

    if lm:
        if isinstance(lm, nn.Module):
            lm_score_impl = rnnlm_score
        else:
            lm_score_impl = ngram_score

    N, _, D_enc = enc_out.shape
    nbest = min(beam, nbest)
    device = enc_out.device
    att_ali = None
    dec_hid = None
    # N x T x F => N*beam x T x F
    enc_out = th.repeat_interleave(enc_out, beam, 0)
    enc_len = th.repeat_interleave(enc_len, beam, 0)
    att_ctx = th.zeros([N * beam, D_enc], device=device)
    proj = th.zeros([N * beam, D_enc], device=device)

    lm_state = None
    beam_param = BeamSearchParam(beam_size=beam,
                                 sos=sos,
                                 eos=eos,
                                 device=device,
                                 penalty=penalty,
                                 coverage=coverage,
                                 lm_weight=lm_weight,
                                 len_norm=len_norm)
    beam_tracker = BatchBeamTracker(N, beam_param)

    # for each utterance
    hypos = [[] for _ in range(N)]
    stop_batch = [False] * N
    # clear states
    att_net.clear()
    # step by step
    for t in range(max_len):
        # N*beam
        pre_out, point = beam_tracker[-1] if t else beam_tracker[0]
        # step forward
        pred, att_ctx, dec_hid, att_ali, proj = decoder.step(att_net,
                                                             pre_out,
                                                             enc_out,
                                                             att_ctx,
                                                             enc_len=enc_len,
                                                             dec_hid=dec_hid,
                                                             att_ali=att_ali,
                                                             proj=proj,
                                                             point=point)
        # compute prob: N*beam x V, nagetive
        am_prob = tf.log_softmax(pred / temperature, dim=-1)

        if lm:
            # beam x V
            lm_prob, lm_state = lm_score_impl(lm, point, pre_out, lm_state)
        else:
            lm_prob = 0

        # local pruning: N*beam x beam
        beam_tracker.prune_beam(am_prob, lm_prob, att_ali=att_ali)

        # process eos nodes
        for u in range(N):
            hyp_ended = beam_tracker.trace_back(u, final=False)
            if hyp_ended:
                hypos[u] += hyp_ended

            if len(hypos[u]) >= beam:
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
