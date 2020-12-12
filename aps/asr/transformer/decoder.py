#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Union, Tuple, Optional, List, Dict
from aps.asr.transformer.embedding import IOEmbedding
from aps.asr.base.attention import padding_mask
from aps.asr.beam_search.lm import adjust_hidden
from aps.const import NEG_INF


def prep_sub_mask(T: int, device: Union[str, th.device] = "cpu") -> th.Tensor:
    """
    Prepare a square sub-sequence masks (-inf/0)
    egs: for N = 8, output
    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    mask = (th.triu(th.ones(T, T, device=device), diagonal=1) == 1).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


def trace_back_hypos(point: th.Tensor,
                     back_point: List[th.Tensor],
                     hist_token: List[th.Tensor],
                     score: th.Tensor,
                     sos: int = 1,
                     eos: int = 2,
                     penalty: float = 0,
                     normalized: bool = True) -> List[Dict]:
    """
    TODO: remove it in the future
    Trace back the decoding transcription sequence from the current time point
    Args:
        point (Tensor): starting point
        back_point (list[Tensor]): father point at each step
        hist_token (list[Tensor]): beam token at each step
        score (Tensor): decoding score
    """
    trans = []
    score = score.tolist()
    for ptr, tok in zip(back_point[::-1], hist_token[::-1]):
        trans.append(tok[point].tolist())
        point = ptr[point]
    hypos = []
    trans = trans[::-1]
    for i, s in enumerate(score):
        token = [t[i] for t in trans]
        score = (s + (len(token) + 1) * penalty) / (len(token) +
                                                    1 if normalized else 1)
        hypos.append({"score": score, "trans": [sos] + token + [eos]})
    return hypos


class TorchTransformerDecoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 vocab_size: int,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0,
                 att_dropout: float = 0.1,
                 pos_enc: bool = True,
                 num_layers: int = 6) -> None:
        super(TorchTransformerDecoder, self).__init__()
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     pos_enc=pos_enc,
                                     scale_embed=scale_embed)
        decoder_layer = TransformerDecoderLayer(att_dim,
                                                nhead,
                                                dim_feedforward=feedforward_dim,
                                                dropout=att_dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(att_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def step(self,
             enc_out: th.Tensor,
             tgt_pad: th.Tensor,
             enc_len: Optional[th.Tensor] = None,
             pre_emb: Optional[th.Tensor] = None,
             out_idx: Optional[int] = None) -> Tuple[th.Tensor]:
        """
        Args:
            enc_out (Tensor): T x N x D
            tgt_pad (Tensor): N x To
            enc_len (Tensor): N or None
            pre_emb (Tensor): T' x N x D
        Return:
            dec_out (Tensor): T+T' x N x D or N x D
        """
        # N x Ti
        memory_mask = None if enc_len is None else (padding_mask(enc_len) == 1)
        if pre_emb is None:
            # genrarte target masks (-inf/0)
            tgt_mask = prep_sub_mask(tgt_pad.shape[-1], device=tgt_pad.device)
            # To+1 x N x E
            tgt_emb = self.tgt_embed(tgt_pad)
        else:
            tgt_mask = prep_sub_mask(tgt_pad.shape[-1] + pre_emb.shape[0],
                                     device=tgt_pad.device)
            # To+1 x N x E
            tgt_emb = self.tgt_embed(tgt_pad, t=pre_emb.shape[0])
            # T x N x E
            tgt_emb = th.cat([pre_emb, tgt_emb], dim=0)
        # To+1 x N x D
        dec_out = self.decoder(tgt_emb,
                               enc_out,
                               tgt_mask=tgt_mask,
                               memory_mask=None,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=memory_mask)
        if out_idx is not None:
            dec_out = dec_out[out_idx]
        # To+1 x N x V
        dec_out = self.output(dec_out)
        return dec_out, tgt_emb

    def forward(self, enc_out: th.Tensor, enc_len: Optional[th.Tensor],
                tgt_pad: th.Tensor) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): T x N x D
            enc_len (Tensor): N or None
            tgt_pad (Tensor): N x To
        Return:
            dec_out (Tensor): T x N x D
        """
        # T x N x V
        dec_out, _ = self.step(enc_out, tgt_pad, enc_len=enc_len)
        return dec_out

    def beam_search(self,
                    enc_out: th.Tensor,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    beam: int = 16,
                    sos: int = -1,
                    eos: int = -1,
                    nbest: int = 8,
                    max_len: int = -1,
                    normalized: bool = True) -> List[Dict]:
        """
        Beam search for Transformer
        Args:
            enc_out = self.encoder(x_emb),  Ti x 1 x D
        """
        if sos < 0 or eos < 0:
            raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
        if max_len <= 0:
            raise RuntimeError(f"Invalid max_len: {max_len:d}")

        nbest = min(beam, nbest)
        if beam > self.vocab_size:
            raise RuntimeError(f"Beam size({beam}) > vocabulary size")

        with th.no_grad():
            # Ti x beam x D
            enc_out = th.repeat_interleave(enc_out, beam, 1)

            device = enc_out.device
            accu_score = th.zeros(beam, device=device)
            hist_token = []
            back_point = []
            lm_state = None

            hypos = []
            dec_seq = []
            point = th.arange(0, beam, dtype=th.int64, device=device)

            # step by step
            for t in range(max_len):
                # beam
                if t:
                    # pre_out: beam x T
                    pre_out = th.tensor(dec_seq, dtype=th.int64, device=device)
                else:
                    # pre_out: beam x 1
                    pre_out = th.tensor([[sos]] * beam,
                                        dtype=th.int64,
                                        device=device)
                # beam x V
                dec_out, _ = self.step(enc_out, pre_out, out_idx=-1)
                # compute prob: beam x V, nagetive
                prob = F.log_softmax(dec_out, dim=-1)

                # add LM score
                if lm:
                    # adjust lm states
                    lm_state = adjust_hidden(point, lm_state)
                    # LM prediction
                    lm_prob, lm_state = lm(pre_out[:, -1:], lm_state)
                    # beam x V
                    prob += F.log_softmax(lm_prob[:, -1], dim=-1) * lm_weight

                # local pruning: beam x beam
                topk_score, topk_token = th.topk(prob, beam, dim=-1)

                if t == 0:
                    # beam
                    accu_score += topk_score[0]
                    token = topk_token[0]
                    dec_seq = token.tolist()
                else:
                    # beam x beam = beam x 1 + beam x beam
                    accu_score = accu_score[..., None] + topk_score
                    # beam*beam => beam
                    accu_score, topk_index = th.topk(accu_score.view(-1),
                                                     beam,
                                                     dim=-1)
                    # point to father's node
                    point = topk_index // beam
                    # beam*beam
                    topk_token = topk_token.view(-1)
                    token = topk_token[topk_index]

                # continue flags
                end_eos = (token == eos).tolist()

                dec_seq = [None for _ in range(beam)]
                # process eos nodes
                if sum(end_eos):
                    idx = [
                        i for i, end_with_eos in enumerate(end_eos)
                        if end_with_eos
                    ]
                    idx = th.tensor(idx, dtype=th.int64, device=device)
                    hyp_full = trace_back_hypos(point[idx],
                                                back_point,
                                                hist_token,
                                                accu_score[idx],
                                                sos=sos,
                                                eos=eos)
                    accu_score[idx] = NEG_INF
                    hypos += hyp_full
                    for i, h in enumerate(hyp_full):
                        dec_seq[idx[i]] = h["trans"]

                if len(hypos) >= beam:
                    break

                # add best token
                hist_token.append(token)
                back_point.append(point)

                # process non-eos nodes
                end_wo_eos = (token != eos).tolist()
                if sum(end_wo_eos):
                    idx = [i for i, go_on in enumerate(end_wo_eos) if go_on]
                    idx = th.tensor(idx, dtype=th.int64, device=device)
                    hyp_partial = trace_back_hypos(idx,
                                                   back_point,
                                                   hist_token,
                                                   accu_score[idx],
                                                   sos=sos,
                                                   eos=eos)
                    # remove fake eos
                    for i, h in enumerate(hyp_partial):
                        dec_seq[idx[i]] = h["trans"][:-1]
                    # process non-eos nodes at the final step
                    if t == max_len - 1:
                        hypos += hyp_partial

            nbest_hypos = sorted(hypos,
                                 key=lambda n: n["score"] /
                                 (len(n["trans"]) - 1 if normalized else 1),
                                 reverse=True)
            return nbest_hypos[:nbest]
