#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from aps.asr.att import AttASR
from aps.asr.base.encoder import TorchRNNEncoder
from aps.asr.enh.mvdr import MvdrBeamformer
from aps.asr.enh.google import CLPFsBeamformer  # same as TimeInvariantEnh
from aps.asr.enh.conv import TimeInvariantEnh, TimeVariantEnh, TimeInvariantAttEnh


class EnhAttASR(nn.Module):
    """
    AttASR with enhancement front-end
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
        super(EnhAttASR, self).__init__()
        # Back-end feature transform
        self.asr_transform = asr_transform
        # LAS-based ASR
        self.las_asr = AttASR(input_size=asr_input_size,
                              vocab_size=vocab_size,
                              eos=eos,
                              sos=sos,
                              ctc=ctc,
                              asr_transform=None,
                              att_type=att_type,
                              att_kwargs=att_kwargs,
                              encoder_type=encoder_type,
                              encoder_proj=encoder_proj,
                              encoder_kwargs=encoder_kwargs,
                              decoder_dim=decoder_dim,
                              decoder_kwargs=decoder_kwargs)
        if asr_cpt:
            las_cpt = th.load(asr_cpt, map_location="cpu")
            self.las_asr.load_state_dict(las_cpt, strict=False)
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
            alis: N x (To+1) x T
            ...
        """
        # mvdr beamforming: N x Ti x F
        x_enh, x_len = self._enhance(x_pad, x_len)
        # outs, alis, ctc_branch, ...
        return self.las_asr(x_enh, x_len, y_pad, ssr=ssr)

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
            return self.las_asr.beam_search(x_enh[0],
                                            beam=beam,
                                            nbest=nbest,
                                            max_len=max_len,
                                            vectorized=vectorized,
                                            normalized=normalized)


class MvdrAttASR(EnhAttASR):
    """
    Mvdr beamformer + Att-based ASR model
    """

    def __init__(
            self,
            enh_input_size=257,
            num_bins=257,
            # beamforming
            enh_transform=None,
            mask_net_kwargs=None,
            mask_net_noise=False,
            mvdr_kwargs=None,
            **kwargs):
        super(MvdrAttASR, self).__init__(**kwargs)
        if enh_transform is None:
            raise RuntimeError("Enhancement feature transform can not be None")
        # Front-end feature extraction
        self.enh_transform = enh_transform
        # TF-mask estimation network
        self.mask_net = TorchRNNEncoder(
            enh_input_size, num_bins * 2 if mask_net_noise else num_bins,
            **mask_net_kwargs)
        self.mask_net_noise = mask_net_noise
        # MVDR beamformer
        self.mvdr_net = MvdrBeamformer(num_bins, **mvdr_kwargs)

    def _enhance(self, x_pad, x_len):
        """
        Mvdr beamforming and asr feature transform
        """
        # mvdr beamforming: N x Ti x F
        x_beam, x_len = self.mvdr_beam(x_pad, x_len)
        # asr feature transform
        x_beam, _ = self.asr_transform(x_beam, None)
        return x_beam, x_len

    def mvdr_beam(self, x_pad, x_len):
        """
        Mvdr beamforming and asr feature transform
        args:
            x_pad: Tensor, N x C x S
            x_len: Tensor, N or None
        """
        # TF-mask
        mask_s, mask_n, x_len, x_cplx = self.pred_mask(x_pad, x_len)
        # mvdr beamforming: N x Ti x F
        x_beam = self.mvdr_net(mask_s, x_cplx, xlen=x_len, mask_n=mask_n)
        return x_beam, x_len

    def pred_mask(self, x_pad, x_len):
        """
        Output TF masks
        args:
            x_pad: Tensor, N x C x S
            x_len: Tensor, N or None
        """
        # enhancement feature transform
        x_pad, x_cplx, x_len = self.enh_transform(x_pad, x_len)
        # TF-mask estimation: N x T x F
        x_mask, x_len = self.mask_net(x_pad, x_len)
        if self.mask_net_noise:
            mask_s, mask_n = th.chunk(x_mask, 2, dim=-1)
        else:
            mask_s, mask_n = x_mask, None
        return mask_s, mask_n, x_len, x_cplx


class BeamAttASR(EnhAttASR):
    """
    Beamformer-based front-end + AttASR
    """

    def __init__(self, mode="tv", enh_transform=None, enh_conf=None, **kwargs):
        super(BeamAttASR, self).__init__(**kwargs)
        conv_enh = {
            "ti": TimeInvariantEnh,
            "tv": TimeVariantEnh,
            "ti_att": TimeInvariantAttEnh,
            "clp": CLPFsBeamformer
        }
        if mode not in conv_enh:
            raise RuntimeError(f"Unknown fs mode: {mode}")
        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        self.enh = conv_enh[mode](**enh_conf)
        self.enh_transform = enh_transform

    def _enhance(self, x_pad, x_len):
        """
        FE processing
        """
        _, x_pad, x_len = self.enh_transform(x_pad, x_len)
        # N x B x T x ...
        x_enh = self.enh(x_pad)
        return x_enh, x_len
