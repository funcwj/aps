#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple
from torch_complex import ComplexTensor
from typing import Optional, List, Dict

from aps.asr.transformers import TransformerASR
from aps.asr.base.encoder import TorchRNNEncoder
from aps.asr.filter.conv import TimeInvariantFilter, TimeVariantFilter, TimeInvariantAttFilter
from aps.asr.filter.mvdr import MvdrBeamformer
from aps.asr.filter.google import CLPFsBeamformer  # same as TimeInvariantEnh


class EnhTransformerASR(nn.Module):
    """
    Transformer with enhancement front-end
    """

    def __init__(
            self,
            asr_input_size: int = 80,
            vocab_size: int = 30,
            sos: int = -1,
            eos: int = -1,
            # feature transform
            asr_transform: Optional[nn.Module] = None,
            asr_cpt: str = "",
            ctc: bool = False,
            encoder_type: str = "transformer",
            encoder_proj: Optional[int] = None,
            encoder_kwargs: Optional[Dict] = None,
            decoder_type: str = "transformer",
            decoder_kwargs: Optional[Dict] = None) -> None:
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
                                              encoder_type=encoder_type,
                                              encoder_proj=encoder_proj,
                                              encoder_kwargs=encoder_kwargs,
                                              decoder_type=decoder_type,
                                              decoder_kwargs=decoder_kwargs)
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

    def forward(
        self,
        x_pad: th.Tensor,
        x_len: Optional[th.Tensor],
        y_pad: th.Tensor,
        ssr: float = 0
    ) -> Tuple[th.Tensor, None, Optional[th.Tensor], Optional[th.Tensor]]:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            ssr: schedule sampling rate
        Return:
            outs: N x (To+1) x V
            ...
        """
        # mvdr beamforming: N x Ti x F
        x_enh, x_len = self._enhance(x_pad, x_len)
        # outs, alis, ctc_branch, ...
        return self.transformer_asr(x_enh, x_len, y_pad, ssr=ssr)

    def beam_search(self,
                    x: th.Tensor,
                    beam: int = 16,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    nbest: int = 8,
                    max_len: int = -1,
                    vectorized: bool = True,
                    normalized: bool = True) -> List[Dict]:
        """
        Args
            x: C x S
        """
        with th.no_grad():
            if x.dim() != 2:
                raise RuntimeError("Now only support for one utterance")
            x_enh, _ = self._enhance(x[None, ...], None)
            return self.transformer_asr.beam_search(x_enh[0],
                                                    beam=beam,
                                                    lm=lm,
                                                    lm_weight=lm_weight,
                                                    nbest=nbest,
                                                    max_len=max_len,
                                                    vectorized=vectorized,
                                                    normalized=normalized)


class BeamTransformerASR(EnhTransformerASR):
    """
    Beamformer-based front-end + LAS ASR
    """

    def __init__(self,
                 mode: str = "tv",
                 enh_transform: Optional[nn.Module] = None,
                 enh_conf: Optional[Dict] = None,
                 **kwargs) -> None:
        super(BeamTransformerASR, self).__init__(**kwargs)
        conv_enh = {
            "ti": TimeInvariantFilter,
            "tv": TimeVariantFilter,
            "ti_att": TimeInvariantAttFilter,
            "clp": CLPFsBeamformer
        }
        if mode not in conv_enh:
            raise RuntimeError(f"Unknown fs mode: {mode}")
        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        self.enh = conv_enh[mode](**enh_conf)
        self.enh_transform = enh_transform

    def _enhance(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        FE processing
        """
        _, x_pad, x_len = self.enh_transform(x_pad, x_len)
        # N x B x T x ...
        x_enh = self.enh(x_pad)
        return x_enh, x_len


class MvdrTransformerASR(EnhTransformerASR):
    """
    Mvdr beamformer + Transformer-based ASR model
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
        super(MvdrTransformerASR, self).__init__(**kwargs)
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

    def _enhance(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Mvdr beamforming and asr feature transform
        Args:
            x_pad: Tensor, N x C x S
            x_len: Tensor, N or None
        """
        # mvdr beamforming: N x Ti x F
        x_beam, x_len = self.mvdr_beam(x_pad, x_len)
        # asr feature transform
        x_beam, _ = self.asr_transform(x_beam, None)
        return x_beam, x_len

    def mvdr_beam(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Mvdr beamforming and asr feature transform
        Args:
            x_pad: Tensor, N x C x S
            x_len: Tensor, N or None
        """
        # TF-mask
        mask_s, mask_n, x_len, x_cplx = self.pred_mask(x_pad, x_len)
        # mvdr beamforming: N x Ti x F
        x_beam = self.mvdr_net(mask_s, x_cplx, xlen=x_len, mask_n=mask_n)
        return x_beam, x_len

    def pred_mask(
        self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor], Optional[th.Tensor],
               ComplexTensor]:
        """
        Output TF masks
        Args:
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
