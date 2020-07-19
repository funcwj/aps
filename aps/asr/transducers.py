#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from aps.asr.transformer.encoder import TorchTransformerEncoder
from aps.asr.transducer.decoder import TorchTransformerDecoder, TorchRNNDecoder
from aps.asr.base.encoder import encoder_instance


class TorchTransducerASR(nn.Module):
    """
    Transducer end-to-end ASR (rnn as decoder)
    """
    def __init__(self,
                 input_size=80,
                 vocab_size=40,
                 blank=-1,
                 asr_transform=None,
                 encoder_type="transformer",
                 encoder_proj=None,
                 encoder_kwargs=None,
                 decoder_kwargs=None):
        super(TorchTransducerASR, self).__init__()
        if blank < 0:
            raise RuntimeError(f"Unsupported blank value: {blank}")
        if encoder_type == "transformer":
            self.encoder = TorchTransformerEncoder(input_size,
                                                   **encoder_kwargs)
            decoder_kwargs["enc_dim"] = encoder_kwargs["att_dim"]
        else:
            if encoder_proj is None:
                raise ValueError("For non-transformer encoder, "
                                 "encoder_proj can not be None")
            self.encoder = encoder_instance(encoder_type, input_size,
                                            encoder_proj, **encoder_kwargs)
            decoder_kwargs["enc_dim"] = encoder_proj
        self.decoder = TorchRNNDecoder(vocab_size, **decoder_kwargs)
        self.blank = blank
        self.asr_transform = asr_transform
        self.encoder_type = encoder_type

    def forward(self, x_pad, x_len, y_pad, y_len):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            y_len: N or None (not used here)
        return:
            dec_out: N x Ti x To+1 x V
        """
        # feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # Ti x N x D or N x Ti x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # Ti x N x D => N x Ti x D
        if self.encoder_type == "transformer":
            enc_out = enc_out.transpose(0, 1)
        # N x Ti x To+1 x V
        dec_out = self.decoder(enc_out, y_pad, blank=self.blank)
        return dec_out, enc_len

    def _dec_prep(self, x):
        """
        Parepare data for decoding
        """
        # raw wave
        if self.asr_transform:
            if x.dim() != 1:
                raise RuntimeError("Now only support for one utterance")
            x, _ = self.asr_transform(x[None, ...], None)
        else:
            # T x F or Beam x T x F
            if x.dim() not in [2, 3]:
                raise RuntimeError(f"Expect 2/3D tensor, but got {x.dim()}")
            x = x[None, ...]
        # Ti x N x D
        enc_out, _ = self.encoder(x, None)
        # Ti x N x D => N x Ti x D
        if self.encoder_type == "transformer":
            enc_out = enc_out.transpose(0, 1)
        return enc_out

    def greedy_search(self, x):
        """
        Beam search for TorchTransducerASR
        """
        with th.no_grad():
            enc_out = self._dec_prep(x)
            return self.decoder.greedy_search(enc_out, blank=self.blank)

    def beam_search(self,
                    x,
                    lm=None,
                    lm_weight=0,
                    beam=16,
                    nbest=8,
                    normalized=True,
                    max_len=-1,
                    vectorized=True):
        """
        Beam search for TorchTransducerASR
        """
        with th.no_grad():
            enc_out = self._dec_prep(x)
            return self.decoder.beam_search(enc_out,
                                            beam=beam,
                                            blank=self.blank,
                                            nbest=nbest,
                                            lm=lm,
                                            lm_weight=lm_weight,
                                            normalized=normalized)


class TransformerTransducerASR(nn.Module):
    """
    Transducer end-to-end ASR (transformer as decoder)
    """
    def __init__(self,
                 input_size=80,
                 vocab_size=40,
                 blank=-1,
                 asr_transform=None,
                 encoder_type="transformer",
                 encoder_proj=None,
                 encoder_kwargs=None,
                 decoder_kwargs=None):
        super(TransformerTransducerASR, self).__init__()
        if blank < 0:
            raise RuntimeError(f"Unsupported blank value: {blank}")
        if encoder_type == "transformer":
            self.encoder = TorchTransformerEncoder(input_size,
                                                   **encoder_kwargs)
        else:
            if encoder_proj is None:
                raise ValueError("For non-transformer encoder, "
                                 "encoder_proj can not be None")
            self.encoder = encoder_instance(encoder_type, input_size,
                                            encoder_proj, **encoder_kwargs)
        decoder_kwargs["enc_dim"] = encoder_proj
        self.decoder = TorchTransformerDecoder(vocab_size, **decoder_kwargs)
        self.blank = blank
        self.asr_transform = asr_transform
        self.encoder_type = encoder_type

    def forward(self, x_pad, x_len, y_pad, y_len):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            y_len: N or None
        return:
            dec_out: N x Ti x To+1 x V
        """
        # feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # Ti x N x D or N x Ti x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # N x Ti x D => Ti x N x D
        if self.encoder_type != "transformer":
            enc_out = enc_out.transpose(0, 1)
        # N x Ti x To+1 x V
        dec_out = self.decoder(enc_out, y_pad, y_len, blank=self.blank)
        return dec_out, enc_len

    def _dec_prep(self, x):
        """
        Prepare data for decoding
        """
        # raw wave
        if self.asr_transform:
            if x.dim() != 1:
                raise RuntimeError("Now only support for one utterance")
            x, _ = self.asr_transform(x[None, ...], None)
        else:
            if x.dim() not in [2, 3]:
                raise RuntimeError(f"Expect 2/3D tensor, but got {x.dim()}")
            x = x[None, ...]
        # Ti x N x D
        enc_out, _ = self.encoder(x, None)
        # N x Ti x D => Ti x N x D
        if self.encoder_type != "transformer":
            enc_out = enc_out.transpose(0, 1)
        return enc_out

    def greedy_search(self, x):
        """
        Greedy search for TransformerTransducerASR
        """
        with th.no_grad():
            enc_out = self._dec_prep(x)
            return self.decoder.greedy_search(enc_out, blank=self.blank)

    def beam_search(self,
                    x,
                    lm=None,
                    lm_weight=0,
                    beam=16,
                    nbest=8,
                    normalized=True,
                    max_len=-1,
                    vectorized=True):
        """
        Beam search for TransformerTransducerASR
        """
        with th.no_grad():
            enc_out = self._dec_prep(x)
            return self.decoder.beam_search(enc_out,
                                            beam=beam,
                                            blank=self.blank,
                                            nbest=nbest,
                                            lm=lm,
                                            lm_weight=lm_weight,
                                            normalized=normalized)
