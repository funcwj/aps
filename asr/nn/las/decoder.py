#!/usr/bin/env python

# wujian@2019

import random

import torch as th
import torch.nn as nn

import torch.nn.functional as F

NEG_INF = th.finfo(th.float32).min


class OneHotEmbedding(nn.Module):
    """
    Onehot encode
    """
    def __init__(self, vocab_size):
        super(OneHotEmbedding, self).__init__()
        self.vocab_size = vocab_size

    def extra_repr(self):
        return f"vocab_size={self.vocab_size}"

    def forward(self, x):
        """
        args:
            x: ...
        return
            e: ... x V
        """
        S = list(x.shape) + [self.vocab_size]
        # ... x V
        H = th.zeros(S, dtype=th.float32, device=x.device)
        # set one
        H = H.scatter(-1, x[..., None], 1)
        return H


class TorchDecoder(nn.Module):
    """
    PyTorch's RNN decoder
    """
    def __init__(self,
                 enc_proj,
                 vocab_size,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0,
                 attention=None,
                 input_feeding=False,
                 vocab_embeded=True):
        super(TorchDecoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        if RNN not in supported_rnn:
            raise RuntimeError(f"unknown RNN type: {RNN}")
        if vocab_embeded:
            self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
            input_size = enc_proj + hidden_size
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
            input_size = enc_proj + vocab_size
        self.decoder = supported_rnn[RNN](input_size,
                                          hidden_size,
                                          num_layers,
                                          batch_first=True,
                                          dropout=dropout,
                                          bidirectional=False)
        self.attend = attention
        self.proj = nn.Linear(hidden_size + enc_proj, enc_proj)
        self.pred = nn.Linear(enc_proj, vocab_size)
        self.input_feeding = input_feeding

    def _step_decoder(self, emb_pre, att_ctx, dec_hid=None):
        """
        args
            emb_pre: N x D_emb
            att_ctx: N x D_enc
        """
        # N x 1 x (D_emb+D_enc)
        dec_in = th.cat([emb_pre, att_ctx], dim=-1).unsqueeze(1)
        # N x 1 x (D_emb+D_enc) => N x 1 x D_dec
        dec_out, hx = self.decoder(dec_in, hx=dec_hid)
        # N x 1 x D_dec => N x D_dec
        return dec_out.squeeze(1), hx

    def _step(self,
              emb_pre,
              enc_out,
              att_ctx,
              dec_hid=None,
              att_ali=None,
              enc_len=None,
              proj=None):
        """
        Make a prediction step
        """
        # dec_out: N x D_dec
        dec_out, dec_hid = self._step_decoder(
            emb_pre, proj if self.input_feeding else att_ctx, dec_hid=dec_hid)
        # att_ali: N x Ti, att_ctx: N x D_enc
        att_ali, att_ctx = self.attend(enc_out, enc_len, dec_out, att_ali)
        # proj: N x D_enc
        proj = self.proj(th.cat([dec_out, att_ctx], dim=-1))
        # pred: N x V
        pred = self.pred(F.relu(proj))
        return att_ali, att_ctx, dec_hid, proj, pred

    def forward(self, enc_pad, enc_len, tgt_pad, sos=-1, schedule_sampling=0):
        """
        args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            tgt_pad: N x To
            schedule_sampling:
                1: using prediction
                0: using ground truth
        return
            outs: N x To x V
            alis: N x To x T
        """
        # reset flags
        self.attend.reset()
        N, _, D_enc = enc_pad.shape
        outs = []  # collect prediction
        att_ali = None  # attention alignments
        dec_hid = None
        dev = enc_pad.device
        # zero init context
        att_ctx = th.zeros([N, D_enc], device=dev)
        proj = th.zeros([N, D_enc], device=dev)
        alis = []  # collect alignments
        # step by step
        #   0   1   2   3   ... T
        # SOS   t0  t1  t2  ... t{T-1}
        #  t0   t1  t2  t3  ... EOS
        for t in range(tgt_pad.shape[-1] + 1):
            # using output at previous time step
            # out: N
            if t and random.random() < schedule_sampling:
                out = th.argmax(outs[-1].detach(), dim=1)
            else:
                if t == 0:
                    out = th.tensor([sos] * N, dtype=th.int64, device=dev)
                else:
                    out = tgt_pad[:, t - 1]
            # N x D_emb or N x V
            emb_pre = self.vocab_embed(out)
            # step forward
            att_ali, att_ctx, dec_hid, proj, pred = self._step(emb_pre,
                                                               enc_pad,
                                                               att_ctx,
                                                               dec_hid=dec_hid,
                                                               att_ali=att_ali,
                                                               enc_len=enc_len,
                                                               proj=proj)
            outs.append(pred)
            alis.append(att_ali)
        # N x To x V
        outs = th.stack(outs, dim=1)
        # N x To x Ti
        alis = th.stack(alis, dim=1)
        return outs, alis

    def beam_search(self,
                    enc_out,
                    beam=8,
                    nbest=1,
                    max_len=None,
                    sos=-1,
                    eos=-1,
                    normalized=True):
        """
        Beam search algothrim (intuitive but not efficient)
        args
            enc_out: 1 x T x F
        """
        # reset flags
        self.attend.reset()

        if beam < nbest:
            raise RuntimeError("N-best value can not exceed " +
                               f"beam size, {beam:d} vs {nbest:d}")
        if sos < 0 or eos < 0:
            raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
        N, T, D_enc = enc_out.shape
        if N != 1:
            raise RuntimeError(
                f"Got batch size {N:d}, now only support one utterance")
        dev = enc_out.device
        att_ctx = th.zeros([N, D_enc], device=dev)
        proj = th.zeros([N, D_enc], device=dev)

        def init_node():
            return {
                "proj": proj,
                "score": 0.0,
                "trans": [sos],
                "att_ali": None,
                "att_ctx": att_ctx,
                "dec_hid": None
            }

        alive = [init_node()]
        hypos = []
        if max_len is None:
            max_len = T
        else:
            max_len = max(T, max_len)

        # step by step
        for t in range(max_len):
            beams = []
            for n in alive:
                # [x], out is different
                out = th.tensor([n["trans"][-1]], dtype=th.int64, device=dev)
                # step forward
                att_ali, att_ctx, dec_hid, proj, pred = self._step(
                    self.vocab_embed(out),
                    enc_out,
                    n["att_ctx"],
                    dec_hid=n["dec_hid"],
                    att_ali=n["att_ali"],
                    proj=n["proj"])
                # compute prob: V, nagetive
                prob = F.log_softmax(pred, dim=1).squeeze(0)
                # beam
                topk_score, topk_index = th.topk(prob, beam)
                # new node
                next_node_templ = {
                    "att_ali": att_ali,
                    "att_ctx": att_ctx,
                    "dec_hid": dec_hid,
                    "score": n["score"],
                    "proj": proj
                }
                for score, index in zip(topk_score, topk_index):
                    # copy
                    new_node = next_node_templ.copy()
                    # add score
                    new_node["score"] += score.item()
                    # add trans
                    new_node["trans"] = n["trans"].copy()
                    new_node["trans"].append(index.item())
                    beams.append(new_node)
            # clip beam
            beams = sorted(beams, key=lambda n: n["score"],
                           reverse=True)[:beam]

            # add finished ones
            hypos.extend([n for n in beams if n["trans"][-1] == eos])
            # keep unfinished ones
            alive = [n for n in beams if n["trans"][-1] != eos]

            if not len(alive):
                break

            if t == max_len - 1:
                for n in alive:
                    n["trans"].append(eos)
                    hypos.append(n)

        # choose nbest
        if normalized:
            nbest_hypos = sorted(hypos,
                                 key=lambda n: n["score"] /
                                 (len(n["trans"]) - 1),
                                 reverse=True)
        else:
            nbest_hypos = sorted(hypos, key=lambda n: n["score"], reverse=True)
        return [{
            "score": n["score"],
            "trans": n["trans"]
        } for n in nbest_hypos[:nbest]]

    def _trace_back_hypos(self,
                          index,
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
            trans.append(tok[index].item())
            index = ptr[index]
        return {"score": score, "trans": [sos] + trans[::-1] + [eos]}

    def beam_search_vectorized(self,
                               enc_out,
                               beam=8,
                               nbest=1,
                               max_len=None,
                               sos=-1,
                               eos=-1,
                               normalized=True):
        """
        Vectorized beam search algothrim (now inferior than normal beam search)
        args
            enc_out: 1 x T x F
        """
        # reset flags
        self.attend.reset()

        if beam < nbest:
            raise RuntimeError("N-best value can not exceed " +
                               f"beam size, {beam:d} vs {nbest:d}")
        if sos < 0 or eos < 0:
            raise RuntimeError(f"Invalid SOS/EOS ID: {sos:d}/{eos:d}")
        N, T, D_enc = enc_out.shape
        if N != 1:
            raise RuntimeError(
                f"Got batch size {N:d}, now only support one utterance")
        if max_len is None:
            max_len = T
        else:
            # if inputs are down-sampled, and small output
            # unit (like graphme) may longer than length of the inputs
            max_len = max(T, max_len)

        dev = enc_out.device
        att_ali = None
        dec_hid = None
        # N x T x F => N*beam x T x F
        enc_out = th.repeat_interleave(enc_out, beam, 0)
        att_ctx = th.zeros([N * beam, D_enc], device=dev)
        proj = th.zeros([N * beam, D_enc], device=dev)

        accu_score = th.zeros(beam, device=dev)
        hist_token = []
        back_point = []

        hypos = []
        # step by step
        for t in range(max_len):
            # beam
            if t:
                out = hist_token[-1]
                point = back_point[-1]
            else:
                out = th.tensor([sos] * (beam * N), dtype=th.int64, device=dev)
                point = th.arange(0, beam, dtype=th.int64, device=dev)

            # swap order
            if dec_hid is not None:
                if isinstance(dec_hid, tuple):
                    # shape: num_layers * num_directions, batch, hidden_size
                    h, c = dec_hid
                    dec_hid = (h[:, point], c[:, point])
                else:
                    dec_hid = dec_hid[:, point]
            if att_ali is not None:
                att_ali = att_ali[point]

            # out_emb: beam x E
            out_emb = self.vocab_embed(out)
            # step forward
            att_ali, att_ctx, dec_hid, proj, pred = self._step(
                out_emb,
                enc_out,
                att_ctx[point],
                dec_hid=dec_hid,
                att_ali=att_ali,
                proj=proj[point])
            # compute prob: beam x V, nagetive
            prob = F.log_softmax(pred, dim=-1)
            # local pruning: beam x beam
            topk_score, topk_token = th.topk(prob, beam, dim=-1)
            if t == 0:
                # beam
                accu_score += topk_score[0]
                token = topk_token[0]
            else:
                # beam x beam = beam x 1 + beam x beam
                accu_score = accu_score[..., None] + topk_score
                # if previous step outputs eos, set -inf
                # then it will not appear after topk operation
                # accu_score[out == eos_dev] = NEG_INF
                # beam*beam => beam
                accu_score, topk_index = th.topk(accu_score.view(-1),
                                                 beam,
                                                 dim=-1)
                # point to father's node
                point = topk_index // beam

                # beam*beam
                topk_token = topk_token.view(-1)
                token = topk_token[topk_index]
                # 1 x beam
                # token = th.gather(topk_token, 1, topk_index[None, ...])
                # token = token[0]

            # continue flags
            not_end = (token != eos).tolist()

            # process eos nodes
            for i, go_on in enumerate(not_end):
                if not go_on:
                    hyp = self._trace_back_hypos(point[i],
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
            # add best token
            hist_token.append(token)
            back_point.append(point)

            # process non-eos nodes at the final step
            if t == max_len - 1:
                for i, go_on in enumerate(not_end):
                    if go_on:
                        hyp = self._trace_back_hypos(i,
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
            nbest_hypos = sorted(hypos, key=lambda n: n["score"], reverse=True)
        return nbest_hypos[:nbest]
