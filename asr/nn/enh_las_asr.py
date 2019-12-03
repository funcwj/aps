#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence

from .las_asr import LasASR
from .las.encoder import TorchEncoder
from .enh.beamformer import MvdrBeamformer


class EnhLasASR(nn.Module):
    """
    Mvdr beamformer + LAS-based ASR model
    """
    def __init__(
            self,
            asr_input_size=80,
            vocab_size=30,
            enh_input_size=257,
            num_bins=257,
            sos=-1,
            eos=-1,
            # feature transform
            asr_transform=None,
            # beamforming
            enh_transform=None,
            mvdr_att_dim=512,
            mask_net_kwargs=None,
            # attention
            att_type="ctx",
            att_kwargs=None,
            # encoder
            encoder_type="common",
            encoder_proj=256,
            encoder_kwargs=None,
            # decoder
            decoder_dim=512,
            decoder_kwargs=None):
        super(EnhLasASR, self).__init__()
        if enh_transform is None:
            raise RuntimeError("Enhancement feature transform can not be None")
        # Front-end feature extraction
        self.enh_transform = enh_transform
        # Back-end feature transform
        self.asr_transform = asr_transform
        # TF-mask estimation network
        self.mask_net = TorchEncoder(enh_input_size, num_bins,
                                     **mask_net_kwargs)
        self.mvdr_net = MvdrBeamformer(num_bins, mvdr_att_dim)
        # LAS-based ASR
        self.las_asr = LasASR(input_size=asr_input_size,
                              vocab_size=vocab_size,
                              eos=eos,
                              sos=sos,
                              asr_transform=None,
                              att_type=att_type,
                              att_kwargs=att_kwargs,
                              encoder_type=encoder_type,
                              encoder_proj=encoder_proj,
                              encoder_kwargs=encoder_kwargs,
                              decoder_dim=decoder_dim,
                              decoder_kwargs=decoder_kwargs)

    def _enhance(self, x_pad, x_len):
        """
        Feature extraction and enhancement
        """
        # N x C x S
        if x_pad.dim() != 3:
            raise RuntimeError(f"Expect 3D tensor, got {x_pad.dim()} instead")
        # enhancement feature transform
        x_pad, x_cplx, x_len = self.enh_transform(x_pad, x_len)
        # TF-mask estimation: N x T x F
        x_mask = self.mask_net(x_pad, x_len)
        # mvdr beamforming: N x Ti x F
        x_beam = self.mvdr_net(x_mask, x_cplx)
        # asr feature transform
        x_beam, x_len = self.asr_transform(x_beam, x_len)
        return x_beam, x_len

    def forward(self, x_pad, x_len, y_pad, ssr=0):
        """
        args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            ssr: schedule sampling rate
        return:
            outs: N x (To+1) x V
            alis: N x (To+1) x T
        """
        # mvdr beamforming: N x Ti x F
        x_beam, x_len = self._enhance(x_pad, x_len)
        # outs, alis
        return self.las_asr(x_beam, x_len, y_pad, ssr=ssr)

    def beam_search(self,
                    x,
                    beam=8,
                    nbest=1,
                    max_len=-1,
                    vectorized=False,
                    normalized=True):
        """
        args
            x: S or Ti x F
        """
        with th.no_grad():
            if x.dim() != 1:
                raise RuntimeError("Now only support for one utterance")
            x_beam, _ = self._enhance(x[None, ...], None)
            return self.las_asr.beam_search(x_beam,
                                            beam=beam,
                                            nbest=nbest,
                                            max_len=max_len,
                                            vectorized=vectorized,
                                            normalized=normalized)
