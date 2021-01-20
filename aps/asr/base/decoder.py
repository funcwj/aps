#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union
from aps.asr.base.layer import OneHotEmbedding, PyTorchRNN

HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]


class PyTorchRNNDecoder(nn.Module):
    """
    PyTorch's RNN decoder
    """

    def __init__(self,
                 enc_proj: int,
                 vocab_size: int,
                 dec_rnn: str = "lstm",
                 rnn_layers: int = 3,
                 rnn_hidden: int = 512,
                 rnn_dropout: float = 0.0,
                 emb_dropout: float = 0.0,
                 input_feeding: bool = False,
                 onehot_embed: bool = False) -> None:
        super(PyTorchRNNDecoder, self).__init__()
        if not onehot_embed:
            self.vocab_embed = nn.Sequential(
                nn.Embedding(vocab_size, rnn_hidden), nn.Dropout(p=emb_dropout))
            input_size = enc_proj + rnn_hidden
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
            input_size = enc_proj + vocab_size
        self.decoder = PyTorchRNN(dec_rnn,
                                  input_size,
                                  rnn_hidden,
                                  rnn_layers,
                                  dropout=rnn_dropout,
                                  bidirectional=False)
        self.proj = nn.Linear(rnn_hidden + enc_proj, enc_proj)
        self.drop = nn.Dropout(p=rnn_dropout)
        self.pred = nn.Linear(enc_proj, vocab_size)
        self.input_feeding = input_feeding
        self.vocab_size = vocab_size

    def step_decoder(
            self,
            emb_pre: th.Tensor,
            att_ctx: th.Tensor,
            dec_hid: Optional[HiddenType] = None
    ) -> Tuple[th.Tensor, HiddenType]:
        """
        Args
            emb_pre: N x D_emb
            att_ctx: N x D_enc
        """
        # N x 1 x (D_emb+D_enc)
        dec_in = th.cat([emb_pre, att_ctx], dim=-1).unsqueeze(1)
        # N x 1 x (D_emb+D_enc) => N x 1 x D_dec
        dec_out, hx = self.decoder(dec_in, hx=dec_hid)
        # N x 1 x D_dec => N x D_dec
        return dec_out.squeeze(1), hx

    def step(
        self,
        att_net: nn.Module,
        out_pre: th.Tensor,
        enc_out: th.Tensor,
        att_ctx: th.Tensor,
        dec_hid: Optional[HiddenType] = None,
        att_ali: Optional[th.Tensor] = None,
        proj: Optional[th.Tensor] = None,
        enc_len: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, HiddenType, th.Tensor, th.Tensor]:
        """
        Make a prediction step
        """
        # N x D_emb or N x V
        emb_pre = self.vocab_embed(out_pre)
        # dec_out: N x D_dec
        if self.input_feeding:
            dec_out, dec_hid = self.step_decoder(emb_pre, proj, dec_hid=dec_hid)
        else:
            dec_out, dec_hid = self.step_decoder(emb_pre,
                                                 att_ctx,
                                                 dec_hid=dec_hid)
        # att_ali: N x Ti, att_ctx: N x D_enc
        att_ali, att_ctx = att_net(enc_out, enc_len, dec_out, att_ali)
        # proj: N x D_enc
        proj = self.proj(th.cat([dec_out, att_ctx], dim=-1))
        proj = self.drop(tf.relu(proj))
        # pred: N x V
        pred = self.pred(proj)
        return pred, att_ctx, dec_hid, att_ali, proj

    def forward(self,
                att_net: nn.Module,
                enc_pad: th.Tensor,
                enc_len: Optional[th.Tensor],
                tgt_pad: th.Tensor,
                schedule_sampling: float = 0) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            tgt_pad: N x To
            schedule_sampling:
                1: using prediction
                0: using ground truth
        Return
            outs: N x To x V
            alis: N x To x T
        """
        N, _, D_enc = enc_pad.shape
        outs = []  # collect prediction
        att_ali = None  # attention alignments
        dec_hid = None
        device = enc_pad.device
        # zero init context
        att_ctx = th.zeros([N, D_enc], device=device)
        proj = th.zeros([N, D_enc], device=device)
        alis = []  # collect alignments
        # step by step
        #   0   1   2   3   ... T
        # SOS   t0  t1  t2  ... t{T-1}
        #  t0   t1  t2  t3  ... EOS
        for t in range(tgt_pad.shape[-1]):
            # using output at previous time step
            # out: N
            if t and random.random() < schedule_sampling:
                tok_pre = th.argmax(outs[-1].detach(), dim=1)
            else:
                tok_pre = tgt_pad[:, t]
            # step forward
            pred, att_ctx, dec_hid, att_ali, proj = self.step(att_net,
                                                              tok_pre,
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
