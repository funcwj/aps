#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Dict, List
from aps.asr.att import AttASR, XfmrASR, NoneOrTensor, AMForwardType
from aps.asr.filter.conv import EnhFrontEnds
from aps.cplx import ComplexTensor
from aps.libs import ApsRegisters


def get_enh_net(enh_type: str,
                enh_kwargs: Dict,
                enh_input_size: Optional[int] = None) -> nn.Module:
    """
    Return enhancement front-end
    """
    if enh_type not in EnhFrontEnds:
        raise ValueError(f"Unknown enhancement front-end: {enh_type}")
    enh_net_cls = EnhFrontEnds[enh_type]
    if enh_type[-4:] == "mvdr":
        if enh_input_size is None:
            enh_input_size = enh_kwargs["num_bins"]
        return enh_net_cls(enh_input_size, **enh_kwargs)
    else:
        return enh_net_cls(**enh_kwargs)


class EnhASRBase(nn.Module):
    """
    Base class for multi-channel enhancement + ASR
    """

    def __init__(
            self,
            asr: nn.Module,
            asr_cpt: str = "",
            enh_input_size: Optional[int] = None,
            # feature transform
            enh_transform: Optional[nn.Module] = None,
            asr_transform: Optional[nn.Module] = None,
            # enhancement
            enh_type: str = "google_clp",
            enh_kwargs: Optional[Dict] = None) -> None:
        super(EnhASRBase, self).__init__()
        # Front-end feature transform
        self.enh_transform = enh_transform
        # Back-end feature transform
        self.asr_transform = asr_transform
        # ASR
        self.asr = asr
        if asr_cpt:
            las_cpt = th.load(asr_cpt, map_location="cpu")
            self.asr.load_state_dict(las_cpt, strict=False)
        # ENH
        self.enh_net = get_enh_net(enh_type,
                                   enh_kwargs,
                                   enh_input_size=enh_input_size)
        self.enh_type = enh_type

    def forward(self,
                x_pad: th.Tensor,
                x_len: NoneOrTensor,
                y_pad: th.Tensor,
                y_len: NoneOrTensor,
                ssr: float = 0) -> AMForwardType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            y_len: N or None, not used here
            ssr: schedule sampling rate
        Return:
            dec_out: N x (To+1) x V
            enc_ctc: N x T x V or None
            enc_len: N or None
        """
        # feature for enhancement
        packed, x_len = self.enh_transform.encode(x_pad, x_len)
        cstft = ComplexTensor(packed[..., 0], packed[..., 1])
        if self.enh_type[-4:] == "mvdr":
            feats = self.enh_transform(packed)
            x_enh = self.enh_net(feats, cstft, inp_len=x_len)
        else:
            x_enh = self.enh_net(cstft)
        # N x T x D, feature for ASR if needed
        if self.asr_transform:
            x_enh, _ = self.asr_transform(x_enh, None)
        # outs, alis, ctc_branch, ...
        return self.asr(x_enh, x_len, y_pad, y_len, ssr=ssr)

    def beam_search(self, x: th.Tensor, **kwargs) -> List[Dict]:
        """
        Args
            x (Tensor): C x S
        """
        with th.no_grad():
            if x.dim() != 2:
                raise RuntimeError("Now only support for one utterance")
            x_enh, _ = self._enhance(x[None, ...], None)
            return self.asr.beam_search(x_enh[0], **kwargs)

    def beam_search_batch(self, batch: List[th.Tensor], **kwargs) -> List[Dict]:
        """
        Args
            batch (list[Tensor]): [C x S, ...]
        """
        with th.no_grad():
            batch_enh = []
            for inp in batch:
                x_enh, _ = self._enhance(inp, None)
                batch_enh.append(x_enh[0])
            return self.asr.beam_search_batch(batch_enh, **kwargs)


@ApsRegisters.asr.register("asr@enh_att")
class EnhAttASR(EnhASRBase):
    """
    AttASR with enhancement front-end
    """

    def __init__(
            self,
            asr_input_size: int = 80,
            enh_input_size: Optional[int] = None,
            vocab_size: int = 30,
            sos: int = -1,
            eos: int = -1,
            ctc: bool = False,
            # feature transform
            enh_transform: Optional[nn.Module] = None,
            asr_transform: Optional[nn.Module] = None,
            # enhancement
            enh_type: str = "google_clp",
            enh_kwargs: Optional[Dict] = None,
            asr_cpt: str = "",
            # attention
            att_type: str = "ctx",
            att_kwargs: Optional[Dict] = None,
            # encoder & decoder
            enc_type: str = "common",
            dec_type: str = "rnn",
            enc_proj: int = 256,
            dec_dim: int = 512,
            enc_kwargs: Optional[Dict] = None,
            dec_kwargs: Optional[Dict] = None) -> None:
        # LAS-based ASR
        las_asr = AttASR(input_size=asr_input_size,
                         vocab_size=vocab_size,
                         eos=eos,
                         sos=sos,
                         ctc=ctc,
                         asr_transform=None,
                         att_type=att_type,
                         att_kwargs=att_kwargs,
                         enc_type=enc_type,
                         enc_proj=enc_proj,
                         enc_kwargs=enc_kwargs,
                         dec_dim=dec_dim,
                         dec_kwargs=dec_kwargs)
        super(EnhAttASR, self).__init__(las_asr,
                                        asr_cpt=asr_cpt,
                                        enh_input_size=enh_input_size,
                                        enh_transform=enh_transform,
                                        asr_transform=asr_transform,
                                        enh_type=enh_type,
                                        enh_kwargs=enh_kwargs)


@ApsRegisters.asr.register("asr@enh_xfmr")
class EnhXfmrASR(EnhASRBase):
    """
    Transformer with enhancement front-end
    """

    def __init__(
            self,
            asr_input_size: int = 80,
            enh_input_size: Optional[int] = None,
            vocab_size: int = 30,
            sos: int = -1,
            eos: int = -1,
            ctc: bool = False,
            # feature transform
            enh_transform: Optional[nn.Module] = None,
            asr_transform: Optional[nn.Module] = None,
            # enhancement
            enh_type: str = "google_clp",
            enh_kwargs: Optional[Dict] = None,
            asr_cpt: str = "",
            # encoder & decoder
            enc_type: str = "xfmr_abs",
            dec_type: str = "xfmr_abs",
            enc_proj: Optional[int] = None,
            enc_kwargs: Optional[Dict] = None,
            dec_kwargs: Optional[Dict] = None) -> None:
        # xfmr based ASR
        transformer_asr = XfmrASR(input_size=asr_input_size,
                                  vocab_size=vocab_size,
                                  sos=sos,
                                  eos=eos,
                                  ctc=ctc,
                                  asr_transform=None,
                                  enc_type=enc_type,
                                  enc_proj=enc_proj,
                                  enc_kwargs=enc_kwargs,
                                  dec_type=dec_type,
                                  dec_kwargs=dec_kwargs)
        super(EnhXfmrASR, self).__init__(transformer_asr,
                                         asr_cpt=asr_cpt,
                                         enh_input_size=enh_input_size,
                                         enh_transform=enh_transform,
                                         asr_transform=asr_transform,
                                         enh_type=enh_type,
                                         enh_kwargs=enh_kwargs)
