#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .transformer.decoder import TorchTransformerDecoder
from .transformer.encoder import TorchTransformerEncoder


class TransformerASR(nn.Module):
    """
    Transformer-based end-to-end ASR
    """
    def __init__(self,
                 input_size=80,
                 vocab_size=40,
                 sos=-1,
                 eos=-1,
                 ctc=False,
                 asr_transform=None,
                 input_embed="conv2d",
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 encoder_layers=6,
                 decoder_layers=6):
        super(TransformerASR, self).__init__()
        if eos < 0 or sos < 0:
            raise RuntimeError(f"Unsupported SOS/EOS value: {sos}/{eos}")
        self.encoder = TorchTransformerEncoder(input_size,
                                               input_embed=input_embed,
                                               att_dim=att_dim,
                                               nhead=nhead,
                                               feedforward_dim=feedforward_dim,
                                               pos_dropout=pos_dropout,
                                               att_dropout=att_dropout,
                                               num_layers=encoder_layers)
        self.decoder = TorchTransformerDecoder(vocab_size,
                                               att_dim=att_dim,
                                               nhead=nhead,
                                               feedforward_dim=feedforward_dim,
                                               pos_dropout=pos_dropout,
                                               att_dropout=att_dropout,
                                               num_layers=decoder_layers)
        self.sos = sos
        self.eos = eos
        self.asr_transform = asr_transform
        # if use CTC, eos & sos should be V and V - 1
        self.ctc = nn.Linear(att_dim, vocab_size -
                             2 if sos != eos else vocab_size -
                             1) if ctc else None

    def forward(self, x_pad, x_len, y_pad, ssr=0):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
        return:
            outs: N x (To+1) x V
        """
        # feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # Ti x N x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # CTC
        if self.ctc:
            ctc_branch = self.ctc(enc_out)
            ctc_branch = ctc_branch.transpose(0, 1)
        else:
            ctc_branch = None
        # To+1 x N x D
        dec_out = self.decoder(enc_out, enc_len, y_pad, sos=self.sos)
        # N x To+1 x D
        dec_out = dec_out.transpose(0, 1).contiguous()
        return dec_out, None, ctc_branch, enc_len

    def beam_search(self,
                    x,
                    beam=16,
                    nbest=8,
                    max_len=-1,
                    vectorized=True,
                    normalized=True):
        """
        Beam search for Transformer
        """
        with th.no_grad():
            # raw wave
            if self.asr_transform:
                if x.dim() != 1:
                    raise RuntimeError("Now only support for one utterance")
                x, _ = self.asr_transform(x[None, ...], None)
            else:
                if x.dim() != 2:
                    raise RuntimeError("Now only support for one utterance")
                x = x[None, ...]
            # Ti x N x D
            enc_out, _ = self.encoder(x, None)
            # beam search
            return self.decoder.beam_search(enc_out,
                                            beam=beam,
                                            sos=self.sos,
                                            eos=self.eos,
                                            nbest=nbest,
                                            max_len=max_len,
                                            vectorized=vectorized,
                                            normalized=normalized)
