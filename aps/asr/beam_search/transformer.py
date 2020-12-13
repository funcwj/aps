#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for transformer based AM (Transformer decoder)
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.beam_search.utils import BeamTracker
from aps.asr.beam_search.lm import lm_score

from typing import List, Dict, Optional


def beam_search(decoder: nn.Module,
                enc_out: th.Tensor,
                lm: Optional[nn.Module] = None,
                lm_weight: float = 0,
                beam: int = 8,
                nbest: int = 1,
                max_len: int = -1,
                sos: int = -1,
                eos: int = -1,
                penalty: float = 0,
                normalized: bool = True,
                temperature: float = 1) -> List[Dict]:
    """
    Vectorized beam search algothrim for transformer decoder
    Args
        enc_out (Tensor): T x 1 x F, encoder output
    """
    if sos < 0 or eos < 0:
        raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
    if max_len <= 0:
        raise RuntimeError(f"Invalid max_len: {max_len:d}")
    _, N, _ = enc_out.shape
    if N != 1:
        raise RuntimeError(
            f"Got batch size {N:d}, now only support one utterance")
    if not hasattr(decoder, "step"):
        raise RuntimeError("Function step should defined in decoder network")
    if beam > decoder.vocab_size:
        raise RuntimeError(f"Beam size({beam}) > vocabulary size")

    nbest = min(beam, nbest)
    device = enc_out.device

    beam_tracker = BeamTracker(beam,
                               sos=sos,
                               eos=eos,
                               device=device,
                               penalty=penalty,
                               normalized=normalized)
    hypos = []
    pre_emb = None
    lm_state = None
    # Ti x beam x D
    enc_out = th.repeat_interleave(enc_out, beam, 1)
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
        prob = tf.log_softmax(dec_out / temperature, dim=-1)

        if lm:
            lm_prob, lm_state = lm_score(lm, point, pre_out, lm_state)
            # beam x V
            prob += lm_prob * lm_weight

        # local pruning
        beam_tracker.prune_beam(prob)
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
