#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings

import torch as th
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence

from typing import Optional, Dict, Tuple, List
from aps.asr.base.encoder import encoder_instance, BaseEncoder
from aps.asr.transformer.encoder import TransformerEncoder
from aps.asr.beam_search.ctc import CtcApi
from aps.libs import ApsRegisters

NoneOrTensor = Optional[th.Tensor]
AMForwardType = Tuple[th.Tensor, th.Tensor, NoneOrTensor]


class ASREncoderBase(nn.Module):
    """
    ASR encoder class
        ctc: whether we use CTC branch
        ead: whether we use encoder & decoder structure
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 ctc: bool = False,
                 ead: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: int = -1,
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(ASREncoderBase, self).__init__()
        assert ctc or ead
        ctc_only = ctc and not ead
        self.vocab_size = vocab_size
        self.asr_transform = asr_transform
        if enc_type in ["xfmr", "cfmr"]:
            self.is_xfmr_encoder = True
            enc_proj = enc_kwargs["arch_kwargs"]["att_dim"]
            enc_kwargs["output_proj"] = vocab_size if ctc_only else -1
            self.encoder = TransformerEncoder(enc_type, input_size,
                                              **enc_kwargs)
        else:
            self.is_xfmr_encoder = False
            self.encoder = encoder_instance(
                enc_type, input_size, vocab_size if ctc_only else enc_proj,
                enc_kwargs, BaseEncoder)
        # for hybrid ctc/aed, we add CTC branch
        self.ctc = nn.Linear(enc_proj, vocab_size) if ead and ctc else None

    def _batch_decoding_prep(self,
                             batch: List[th.Tensor],
                             batch_first: bool = True) -> Tuple[th.Tensor]:
        """
        Get encoder output for the batch decoding
        """
        # raw wave
        if len(batch) == 1:
            warnings.warn("Got one utterance, use beam_search (...) instead")
        # NOTE: If we do zero padding on the input features/signals and form them as a batch,
        #       the output may slightly differ with the non-padding version. Thus we use for loop here
        outs = []
        for inp in batch:
            if self.asr_transform:
                inp, _ = self.asr_transform(inp[None, ...], None)
            else:
                inp = inp[None, ...]
            # N x Ti x D
            enc_out, _ = self.encoder(inp, None)
            outs.append(enc_out[0])

        lens = [out.shape[0] for out in outs]
        # T x N x D
        enc_out = pad_sequence(outs, batch_first=False)
        enc_len = th.tensor(lens, device=enc_out.device)
        # enc_out: N x T x D or T x N x D
        return enc_out.transpose(0, 1) if batch_first else enc_out, enc_len

    def _decoding_prep(self,
                       x: th.Tensor,
                       batch_first: bool = True) -> th.Tensor:
        """
        Get encoder output for ASR decoding
        """
        x_dim = x.dim()
        # raw waveform or feature
        if self.asr_transform:
            if x_dim not in [1, 2]:
                raise RuntimeError(
                    "Expect 1/2D (single/multi-channel waveform or single " +
                    f"channel feature) tensor, but get {x_dim}")
            # 1 x C x T x ... or 1 x T x F
            x, _ = self.asr_transform(x[None, ...], None)
        # already feature
        else:
            if x_dim not in [2, 3]:
                raise RuntimeError(
                    "Expect 2/3D (single or multi-channel waveform) " +
                    f"tensor, but got {x_dim}")
            x = x[None, ...]
        # N x Ti x D
        enc_out, _ = self.encoder(x, None)
        # N x Ti x D or Ti x N x D (for xfmr)
        return enc_out if batch_first else enc_out.transpose(0, 1)

    def _training_prep(self, x_pad: th.Tensor,
                       x_len: NoneOrTensor) -> AMForwardType:
        """
        Get encoder output for AM training
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
        Return:
            enc_out: N x Ti x D
            enc_ctc: N x Ti x V or None
            enc_len: N or None
        """
        # asr feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # N x Ti x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # CTC branch
        enc_ctc = enc_out
        if self.ctc:
            enc_ctc = self.ctc(enc_out)
        return enc_out, enc_ctc, enc_len


@ApsRegisters.asr.register("asr@ctc")
class CtcASR(ASREncoderBase):
    """
    A simple ASR encoder structure trained with CTC loss
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 30,
                 ctc: bool = True,
                 ead: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: int = -1,
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(CtcASR, self).__init__(input_size,
                                     vocab_size,
                                     ctc=ctc,
                                     ead=ead,
                                     asr_transform=asr_transform,
                                     enc_type=enc_type,
                                     enc_proj=enc_proj,
                                     enc_kwargs=enc_kwargs)

    def forward(self, x_pad: th.Tensor, x_len: NoneOrTensor) -> AMForwardType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
        Return:
            enc_ctc: N x T x V
            enc_len: N or None
        """
        return self._training_prep(x_pad, x_len)

    def beam_search(self, x: th.Tensor, **kwargs) -> List[Dict]:
        """
        CTC beam search if has CTC branch
        Args
            x (Tensor): audio samples or acoustic features, S or Ti x F
        """
        ctc_api = CtcApi(self.vocab_size - 1)
        with th.no_grad():
            # N x T x D
            enc_out = self._decoding_prep(x, batch_first=True)
            if self.ctc is not None:
                enc_out = self.ctc(enc_out)
            return ctc_api.beam_search(enc_out[0], **kwargs)

    def ctc_align(self, x: th.Tensor, y: th.Tensor) -> Dict:
        """
        Do CTC viterbi align if has CTC branch
        Args:
            x (Tensor): audio samples or acoustic features, S or Ti x F
            y (Tensor): reference sequence, U
        """
        ctc_api = CtcApi(self.vocab_size - 1)
        with th.no_grad():
            # N x T x D
            enc_out = self._decoding_prep(x, batch_first=True)
            if self.ctc is not None:
                enc_out = self.ctc(enc_out)
            return ctc_api.viterbi_align(enc_out[0], y)
