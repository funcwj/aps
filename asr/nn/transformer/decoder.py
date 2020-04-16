#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

import torch.nn.functional as F

try:
    from torch.nn import TransformerDecoder, TransformerDecoderLayer
except:
    raise ImportError("import Transformer module failed")

from .embedding import IOEmbedding
from ..las.attention import padding_mask
from ..las.decoder import NEG_INF

IGNORE_ID = -1


def prep_sub_mask(T, device="cpu"):
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


class TorchTransformerDecoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """
    def __init__(self,
                 vocab_size,
                 att_dim=512,
                 enc_dim=None,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0,
                 att_dropout=0.1,
                 num_layers=6):
        super(TorchTransformerDecoder, self).__init__()
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout)
        decoder_layer = TransformerDecoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        if enc_dim and enc_dim != att_dim:
            self.enc_proj = nn.Linear(enc_dim, att_dim)
        else:
            self.enc_proj = None
        self.output = nn.Linear(att_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(self, enc_out, enc_len, tgt_pad, sos=-1):
        """
        args:
            enc_out: Ti x N x D
            enc_len: N or None
            tgt_pad: N x To
        return:
            dec_out: To+1 x N x D
        """
        if sos < 0:
            raise ValueError(f"Invalid sos value: {sos}")
        # N x Ti
        memory_mask = None if enc_len is None else (padding_mask(enc_len) == 1)
        # N x To+1
        tgt_pad = F.pad(tgt_pad, (1, 0), value=sos)
        # genrarte target masks (-inf/0)
        tgt_mask = prep_sub_mask(tgt_pad.shape[-1], device=tgt_pad.device)
        # To+1 x N x E
        tgt_pad = self.tgt_embed(tgt_pad)
        # Ti x N x D
        if self.enc_proj:
            enc_out = self.enc_proj(enc_out)
        # To+1 x N x D
        dec_out = self.decoder(tgt_pad,
                               enc_out,
                               tgt_mask=tgt_mask,
                               memory_mask=None,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=memory_mask)
        # To+1 x N x V
        dec_out = self.output(dec_out)
        return dec_out

    def beam_search(self,
                    enc_out,
                    beam=16,
                    sos=-1,
                    eos=-1,
                    nbest=8,
                    max_len=-1,
                    vectorized=True,
                    normalized=True):
        """
        Beam search for Transformer
        args:
            enc_out = self.encoder(x_emb),  Ti x 1 x D
        """
        def _trace_back_hypos(point,
                              back_point,
                              hist_token,
                              score,
                              sos=1,
                              eos=2):
            """
            Trace back from current time point
            """
            trans = []
            score = score.item()
            for ptr, tok in zip(back_point[::-1], hist_token[::-1]):
                trans.append(tok[point].item())
                point = ptr[point]
            return {"score": score, "trans": [sos] + trans[::-1] + [eos]}

        if sos < 0 or eos < 0:
            raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")

        T, _, _ = enc_out.shape
        if max_len <= 0:
            max_len = T
        else:
            max_len = max(T, max_len)

        nbest = min(beam, nbest)
        if beam > self.vocab_size:
            raise RuntimeError(f"Beam size({beam}) > vocabulary size")

        with th.no_grad():
            # Ti x N x D
            if self.enc_proj:
                enc_out = self.enc_proj(enc_out)
            # Ti x beam x D
            enc_out = th.repeat_interleave(enc_out, beam, 1)

            dev = enc_out.device
            accu_score = th.zeros(beam, device=dev)
            hist_token = []
            back_point = []
            pre_emb = None

            hypos = []
            # step by step
            for t in range(max_len):
                # target mask
                tgt_mask = prep_sub_mask(t + 1, device=dev)
                # beam
                if t:
                    point = back_point[-1]
                    cur_emb = self.tgt_embed(hist_token[-1][point][..., None],
                                             t=t)
                    pre_emb = th.cat([pre_emb, cur_emb], dim=0)
                else:
                    point = th.arange(0, beam, dtype=th.int64, device=dev)
                    out = th.tensor([sos] * beam, dtype=th.int64, device=dev)
                    # 1 x beam x E
                    pre_emb = self.tgt_embed(out[..., None])
                # Tcur - 1 x beam x D
                dec_out = self.decoder(pre_emb, enc_out, tgt_mask=tgt_mask)[-1]
                # beam x V
                dec_out = self.output(dec_out)
                # compute prob: beam x V, nagetive
                prob = F.log_softmax(dec_out, dim=-1)
                # local pruning: beam x beam
                topk_score, topk_token = th.topk(prob, beam, dim=-1)

                if t == 0:
                    # beam
                    accu_score += topk_score[0]
                    token = topk_token[0]
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
                not_end = (token != eos).tolist()

                # process eos nodes
                for i, go_on in enumerate(not_end):
                    if not go_on:
                        hyp = _trace_back_hypos(point[i],
                                                back_point,
                                                hist_token,
                                                accu_score[i],
                                                sos=sos,
                                                eos=eos)
                        accu_score[i] = NEG_INF
                        hypos.append(hyp)

                # all True
                if sum(not_end) == 0:
                    break

                if len(hypos) >= beam:
                    break

                # add best token
                hist_token.append(token)
                back_point.append(point)

                # process non-eos nodes at the final step
                if t == max_len - 1:
                    for i, go_on in enumerate(not_end):
                        if go_on:
                            hyp = _trace_back_hypos(i,
                                                    back_point,
                                                    hist_token,
                                                    accu_score[i],
                                                    sos=sos,
                                                    eos=eos)
                            hypos.append(hyp)
            if normalized:
                nbest_hypos = sorted(hypos,
                                     key=lambda n: n["score"] /
                                     (len(n["trans"]) - 1),
                                     reverse=True)
            else:
                nbest_hypos = sorted(hypos,
                                     key=lambda n: n["score"],
                                     reverse=True)
            return nbest_hypos[:nbest]