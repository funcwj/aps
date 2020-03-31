#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

import torch.nn.functional as F

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from ..transformer.embedding import IOEmbedding
from ..transformer.decoder import prep_sub_mask
from ..las.attention import padding_mask
from ..las.decoder import OneHotEmbedding
from ..lm.rnn import TorchLM

IGNORE_ID = -1


class TorchRNNDecoder(nn.Module):
    """
    Wrapper for pytorch's RNN Decoder
    """
    def __init__(self,
                 vocab_size,
                 embed_size=512,
                 enc_dim=512,
                 jot_dim=512,
                 dec_rnn="lstm",
                 dec_layers=3,
                 dec_hidden=512,
                 dec_dropout=0.0):
        super(TorchRNNDecoder, self).__init__()
        RNN = dec_rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if embed_size != vocab_size:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        # uni-dir RNNs
        self.decoder = supported_rnn[RNN](embed_size,
                                          dec_hidden,
                                          dec_layers,
                                          batch_first=True,
                                          dropout=dec_dropout,
                                          bidirectional=False)
        self.enc_proj = nn.Linear(enc_dim, jot_dim, bias=False)
        self.dec_proj = nn.Linear(dec_hidden, jot_dim)
        self.vocab_size = vocab_size
        self.output = nn.Linear(jot_dim, vocab_size, bias=False)

    def forward(self, enc_out, tgt_pad, sos=-1):
        """
        args:
            enc_out: N x Ti x D
            tgt_pad: N x To
            tgt_len: N or None
        return:
            output: N x Ti x To+1 x V
        """
        if sos < 0:
            raise ValueError(f"Invalid sos value: {sos}")
        # N x To+1
        tgt_pad = F.pad(tgt_pad, (1, 0), value=sos)
        # N x To+1 x E
        tgt_pad = self.vocab_embed(tgt_pad)
        # N x To+1 x D
        dec_out, _ = self.decoder(tgt_pad)
        # N x Ti x J
        enc_out = self.enc_proj(enc_out)
        # N x To+1 x J
        dec_out = self.dec_proj(dec_out)
        # N x Ti x To+1 x J
        add_out = th.tanh(enc_out[..., None, :] + dec_out[:, None])
        # N x Ti x To+1 x V
        return self.output(add_out)


class TorchTransformerDecoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """
    def __init__(self,
                 vocab_size,
                 enc_dim=None,
                 jot_dim=512,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 num_layers=6):
        super(TorchTransformerDecoder, self).__init__()
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout)
        decoder_layer = TransformerEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout)
        self.decoder = TransformerEncoder(decoder_layer, num_layers)
        self.enc_proj = nn.Linear(enc_dim if enc_dim else att_dim,
                                  jot_dim,
                                  bias=False)
        self.dec_proj = nn.Linear(att_dim, jot_dim)
        self.vocab_size = vocab_size
        self.output = nn.Linear(jot_dim, vocab_size, bias=False)

    def forward(self, enc_out, tgt_pad, tgt_len, sos=-1):
        """
        args:
            enc_out: Ti x N x D
            tgt_pad: N x To
            tgt_len: N or None
        return:
            output: N x Ti x To+1 x V
        """
        if sos < 0:
            raise ValueError(f"Invalid sos value: {sos}")
        # N x Ti
        pad_mask = None if tgt_len is None else (padding_mask(tgt_len) == 1)
        # N x To+1
        tgt_pad = F.pad(tgt_pad, (1, 0), value=sos)
        # genrarte target masks (-inf/0)
        tgt_mask = prep_sub_mask(tgt_pad.shape[-1], device=tgt_pad.device)
        # To+1 x N x E
        tgt_pad = self.tgt_embed(tgt_pad)
        # To+1 x N x D
        dec_out = self.decoder(tgt_pad,
                               mask=tgt_mask,
                               src_key_padding_mask=pad_mask)
        enc_out = self.enc_proj(enc_out)
        dec_out = self.dec_proj(dec_out)
        # To+1 x Ti x N x J
        add_out = th.tanh(enc_out[None, ...] + dec_out[:, None])
        # To+1 x Ti x N x J
        output = self.output(add_out)
        # N x Ti x To+1 x J
        return output.transpose(0, 2).contiguous()