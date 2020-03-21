#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from .transformer_asr import TransformerASR
from .enh.conv import TimeInvariantFE, TimeVariantFE


class EnhTransformerASR(nn.Module):
    """
    Transformer with enhancement front-end
    """
    def __init__(
            self,
            asr_input_size=80,
            vocab_size=30,
            sos=-1,
            eos=-1,
            # feature transform
            asr_transform=None,
            asr_cpt="",
            ctc=False,
            input_embed="conv2d",
            att_dim=512,
            nhead=8,
            feedforward_dim=1024,
            pos_dropout=0.1,
            att_dropout=0.1,
            encoder_layers=6,
            decoder_layers=6):
        super(EnhTransformerASR, self).__init__()
        # Back-end feature transform
        self.asr_transform = asr_transform
        # LAS-based ASR
        self.transformer_asr = TransformerASR(input_size=asr_input_size,
                                              vocab_size=vocab_size,
                                              sos=sos,
                                              eos=eos,
                                              ctc=ctc,
                                              asr_transform=None,
                                              input_embed=input_embed,
                                              att_dim=att_dim,
                                              nhead=nhead,
                                              feedforward_dim=feedforward_dim,
                                              pos_dropout=pos_dropout,
                                              att_dropout=att_dropout,
                                              encoder_layers=encoder_layers,
                                              decoder_layers=decoder_layers)
        if asr_cpt:
            transformer_cpt = th.load(asr_cpt, map_location="cpu")
            self.transformer_asr.load_state_dict(transformer_cpt, strict=False)
        self.sos = sos
        self.eos = eos

    def _enhance(self, x_pad, x_len):
        """
        Enhancement and asr feature transform
        """
        raise NotImplementedError

    def forward(self, x_pad, x_len, y_pad, ssr=0):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            ssr: schedule sampling rate
        return:
            outs: N x (To+1) x V
            ...
        """
        # mvdr beamforming: N x Ti x F
        x_enh, x_len = self._enhance(x_pad, x_len)
        # outs, alis, ctc_branch, ...
        return self.transformer_asr(x_enh, x_len, y_pad, ssr=ssr)

    def beam_search(self,
                    x,
                    beam=8,
                    nbest=1,
                    max_len=-1,
                    vectorized=False,
                    normalized=True):
        """
        args
            x: C x S
        """
        with th.no_grad():
            if x.dim() != 2:
                raise RuntimeError("Now only support for one utterance")
            x_enh, _ = self._enhance(x[None, ...], None)
            return self.transformer_asr.beam_search(x_enh[0],
                                                    beam=beam,
                                                    nbest=nbest,
                                                    max_len=max_len,
                                                    vectorized=vectorized,
                                                    normalized=normalized)


class ConvFeTransformerASR(EnhTransformerASR):
    """
    Convolution-based front-end + Transformer ASR
    """
    def __init__(self, mode="tv", enh_transform=None, fe_conf=None, **kwargs):
        super(ConvFeTransformerASR, self).__init__(**kwargs)
        conv_fe = {"ti": TimeInvariantFE, "tv": TimeVariantFE}
        if mode not in conv_fe:
            raise RuntimeError(f"Unknown fs mode: {mode}")
        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        self.fe = conv_fe[mode](**fe_conf)
        self.enh_transform = enh_transform

    def _enhance(self, x_pad, x_len):
        """
        Front-end processing
        """
        _, x_pad, x_len = self.enh_transform(x_pad, x_len)
        # N x T x ...
        x_enh = self.fe(x_pad)
        return x_enh, x_len