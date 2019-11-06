#!/usr/bin/env python

# wujian@2019

import random

import torch as th
import torch.nn as nn

import torch.nn.functional as F


class TorchDecoder(nn.Module):
    """
    PyTorch's RNN decoder
    """
    def __init__(self,
                 input_size,
                 vocab_size,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0,
                 attention=None):
        super(TorchDecoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
        if RNN not in supported_rnn:
            raise RuntimeError("unknown RNN type: {}".format(RNN))
        self.decoder = supported_rnn[RNN](input_size,
                                          hidden_size,
                                          num_layers,
                                          batch_first=True,
                                          dropout=dropout,
                                          bidirectional=False)
        self.attend = attention
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, vocab_size))

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
        dec_out = None
        # zero init context
        att_ctx = th.zeros([N, D_enc], device=enc_pad.device)
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
                    out = th.tensor([sos] * N,
                                    dtype=th.int64,
                                    device=enc_pad.device)
                else:
                    out = tgt_pad[:, t - 1]
            # N x D_emb
            emb_pre = self.vocab_embed(out)
            # dec_out: N x D_dec
            dec_out, dec_hid = self._step_decoder(emb_pre,
                                                  att_ctx,
                                                  dec_hid=dec_hid)
            # att_ali: N x Ti
            # att_ctx: N x D_enc
            att_ali, att_ctx = self.attend(enc_pad, enc_len, dec_out, att_ali)
            # pred: N x V
            pred = self.output(th.cat([dec_out, att_ctx], dim=-1))
            # pred = self.output(dec_out)
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
                    nbest=5,
                    max_len=None,
                    sos=-1,
                    eos=-1):
        """
        Beam search algothrim
        args
            enc_out: 1 x T x F
        """
        # reset flags
        self.attend.reset()

        if sos < 0 or eos < 0:
            raise RuntimeError("Invalid SOS/EOS ID: {:d}/{:d}".format(
                sos, eos))
        N, T, D_enc = enc_out.shape
        if N != 1:
            raise RuntimeError("Got batch size {:d}, now only "
                               "support one utterance".format(N))
        att_ctx = th.zeros([N, D_enc], device=enc_out.device)

        def init_node():
            return {
                "score": 0.0,
                "trans": [sos],
                "att_ali": None,
                "att_ctx": att_ctx,
                "dec_hid": None
            }

        alive = [init_node()]
        ended = []
        if max_len is None:
            max_len = T

        # step by step
        for t in range(max_len):
            beams = []
            for n in alive:
                # [x], out is different
                out = th.tensor([n["trans"][-1]],
                                dtype=th.int64,
                                device=enc_out.device)
                # dec_out: 1 x D_dec
                dec_out, dec_hid = self._step_decoder(self.vocab_embed(out),
                                                      n["att_ctx"],
                                                      dec_hid=n["dec_hid"])
                # compute align context, 1 x D_enc
                att_ali, att_ctx = self.attend(enc_out, None, dec_out,
                                               n["att_ali"])
                # pred: 1 x V
                pred = self.output(th.cat([dec_out, att_ctx], dim=-1))
                # compute prob: V, nagetive
                prob = F.log_softmax(pred, dim=1).squeeze(0)
                # beam
                best_score, best_index = th.topk(prob, beam)
                # new node
                next_node_templ = {
                    "att_ali": att_ali,
                    "att_ctx": att_ctx,
                    "dec_hid": dec_hid,
                    "score": n["score"],
                }
                for c in range(beam):
                    # copy
                    new_node = next_node_templ.copy()
                    # add score
                    new_node["score"] += best_score[c].item()
                    # add trans
                    new_node["trans"] = n["trans"].copy()
                    new_node["trans"].append(best_index[c].item())
                    beams.append(new_node)
                # clip beam
                beams = sorted(beams, key=lambda n: n["score"],
                               reverse=True)[:beam]

            if t == max_len - 1:
                for n in beams:
                    n["trans"].append(eos)

            # add finished ones
            ended.extend([n for n in beams if n["trans"][-1] == eos])
            # keep unfinished ones
            alive = [n for n in beams if n["trans"][-1] != eos]

            if not len(alive):
                break
        # choose nbest
        nbest_nodes = sorted(ended, key=lambda n: n["score"],
                             reverse=True)[:nbest]
        return [{
            "score": n["score"],
            "trans": n["trans"]
        } for n in nbest_nodes]
