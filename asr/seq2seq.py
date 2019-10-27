#!/usr/bin/env python

# wujian@2019

import random

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from nn.decoder import TorchDecoder
from nn.encoder import encoder_instance
from nn.attention import att_instance


class Seq2Seq(nn.Module):
    """
    A simple attention based sequence-to-sequence model
    """
    def __init__(
            self,
            input_size=80,
            vocab_size=30,
            sos=-1,
            eos=-1,
            att_type="ctx",
            att_kwargs={"att_dim": 512},
            # encoder
            encoder_type="common",
            encoder_dim=512,
            encoder_proj=256,
            encoder_layers=3,
            encoder_rnn="lstm",
            encoder_dropout=0.0,
            # decoder
            decoder_dim=512,
            decoder_layers=2,
            decoder_dropout=0.0,
            decoder_rnn="lstm"):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder_instance(encoder_type,
                                        input_size,
                                        encoder_proj,
                                        rnn=encoder_rnn,
                                        bidirectional=True,
                                        dropout=encoder_dropout,
                                        hidden_size=encoder_dim)
        attend = att_instance(att_type, encoder_proj, decoder_dim,
                              **att_kwargs)
        self.decoder = TorchDecoder(encoder_proj + decoder_dim,
                                    vocab_size,
                                    rnn=decoder_rnn,
                                    hidden_size=decoder_dim,
                                    dropout=decoder_dropout,
                                    num_layers=decoder_layers,
                                    attention=attend)
        if not eos or not sos:
            raise RuntimeError("Unsupported SOS/EOS "
                               "value: {:d}/{:d}".format(sos, eos))
        self.sos = sos
        self.eos = eos

    def forward(self, x_pad, x_len, y_pad, ssr=0):
        """
        args:
            x_pad: N x Ti x D
            x_len: N or None
            y_pad: N x To
            ssr: schedule sampling rate
        return:
            outs: N x (To+1) x V
            alis: N x (To+1) x T
        """
        # N x Ti x D
        enc_out = self.encoder(x_pad, x_len)
        # N x (To+1), pad SOS
        outs, alis = self.decoder(enc_out,
                                  x_len,
                                  y_pad,
                                  sos=self.sos,
                                  schedule_sampling=ssr)
        return outs, alis

    def beam_search(self, x, beam=8, nbest=5, max_len=None):
        """
        args
            x: Ti x F
        """
        if x.dim() != 2:
            raise RuntimeError("Now only support for one utterance")

        with th.no_grad():
            # 1 x Ti x F
            enc_out = self.encoder(x.unsqueeze(0), None)
            return self.decoder.beam_search(enc_out,
                                            beam=beam,
                                            nbest=nbest,
                                            max_len=max_len,
                                            sos=self.sos,
                                            eos=self.eos)


import matplotlib.pyplot as plt


def foo():
    N, V, F = 10, 30, 80
    nnet_conf = {
        "input_size": F,
        "sos": 1,
        "eos": 2,
        "vocab_size": V,
        "encoder_type": "common",
        "encoder_dim": 128,
        "encoder_proj": 128,
        "encoder_layers": 2,
        "encoder_rnn": "lstm",
        "encoder_dropout": 0.0,
        "decoder_dim": 128,
        "decoder_layers": 2,
        "decoder_dropout": 0.0,
        "decoder_rnn": "lstm",
        "att_type": "loc",
        "att_kwargs": {
            "att_dim": 128,
            "att_kernel": 15,
            "att_channels": 128
        }
    }
    # 1.5s
    S, L = 50, 30
    nnet = Seq2Seq(**nnet_conf)

    x_len = th.randint(S, S * 2, (N, ))
    S = x_len.max().item()

    x_len, _ = th.sort(x_len, descending=True)

    x_pad = th.rand(N, S, F) * 10
    y_pad = th.randint(0, V, (N, L))

    print(x_pad.shape)
    print(y_pad.shape)
    # print(y_pad)
    outs, alis = nnet(x_pad, x_len, y_pad, ssr=0)
    print(outs.shape)
    print(alis.shape)
    plt.imshow(alis[5].detach().numpy())
    plt.show()


if __name__ == "__main__":
    foo()