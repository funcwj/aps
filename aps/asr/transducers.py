#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Dict, List
from aps.asr.ctc import ASREncoderBase, NoneOrTensor, AMForwardType
from aps.asr.transducer.decoder import TorchTransformerDecoder, PyTorchRNNDecoder
from aps.asr.beam_search.transducer import greedy_search, beam_search
from aps.libs import ApsRegisters


class ASRTransducerBase(ASREncoderBase):
    """
    Base class for Transducer ASR
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 blank: int = -1,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "xfmr",
                 enc_proj: int = -1,
                 enc_kwargs: Dict = {}) -> None:
        super(ASRTransducerBase, self).__init__(input_size,
                                                vocab_size,
                                                ctc=False,
                                                asr_transform=asr_transform,
                                                enc_type=enc_type,
                                                enc_proj=enc_proj,
                                                enc_kwargs=enc_kwargs)
        if blank < 0:
            raise RuntimeError(f"Unsupported blank value: {blank}")
        self.blank = blank
        self.decoder = None

    def greedy_search(self, x: th.Tensor) -> List[Dict]:
        """
        Greedy search for TransducerASR
        """
        with th.no_grad():
            enc_out = self._decoding_prep(x)
            return greedy_search(self.decoder, enc_out, blank=self.blank)

    def beam_search(self,
                    x: th.Tensor,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    beam_size: int = 16,
                    nbest: int = 8,
                    len_norm: bool = True,
                    max_len: int = -1,
                    **kwargs) -> List[Dict]:
        """
        Beam search for TransducerASR
        """
        with th.no_grad():
            enc_out = self._decoding_prep(x)
            return beam_search(self.decoder,
                               enc_out,
                               beam_size=beam_size,
                               blank=self.blank,
                               nbest=nbest,
                               lm=lm,
                               lm_weight=lm_weight,
                               len_norm=len_norm)


@ApsRegisters.asr.register("asr@transducer")
class TransducerASR(ASRTransducerBase):
    """
    Transducer based ASR model with (Non-)Transformer encoder + RNN decoder
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 40,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "xfmr",
                 enc_proj: int = -1,
                 dec_type: str = "rnn",
                 enc_kwargs: Dict = None,
                 dec_kwargs: Dict = None) -> None:
        super(TransducerASR, self).__init__(input_size=input_size,
                                            vocab_size=vocab_size,
                                            blank=vocab_size - 1,
                                            asr_transform=asr_transform,
                                            enc_type=enc_type,
                                            enc_proj=enc_proj,
                                            enc_kwargs=enc_kwargs)
        if dec_type != "rnn":
            raise ValueError("TransducerASR: the decoder must be rnn")
        if self.is_xfmr_encoder:
            dec_kwargs["enc_dim"] = enc_kwargs["arch_kwargs"]["att_dim"]
        else:
            dec_kwargs["enc_dim"] = enc_proj
        self.decoder = PyTorchRNNDecoder(vocab_size, **dec_kwargs)

    def forward(self, x_pad: th.Tensor, x_len: NoneOrTensor, y_pad: th.Tensor,
                y_len: NoneOrTensor) -> AMForwardType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            y_len: N or None (not used here)
        Return:
            enc_out: N x Ti x D
            dec_out: N x Ti x To+1 x V
            enc_len: N
        """
        # go through feature extractor & encoder
        enc_out, _, enc_len = self._training_prep(x_pad, x_len)
        # N x To+1
        tgt_pad = tf.pad(y_pad, (1, 0), value=self.blank)
        # N x Ti x To+1 x V
        dec_out = self.decoder(enc_out, tgt_pad)
        return enc_out, dec_out, enc_len


@ApsRegisters.asr.register("asr@xfmr_transducer")
class XfmrTransducerASR(ASRTransducerBase):
    """
    Transducer based ASR model with (Non-)Transformer encoder + Transformer decoder
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 40,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "xfmr",
                 enc_proj: Optional[int] = None,
                 enc_kwargs: Dict = {},
                 dec_type: str = "xfmr",
                 dec_kwargs: Dict = {}) -> None:
        super(XfmrTransducerASR, self).__init__(input_size=input_size,
                                                vocab_size=vocab_size,
                                                blank=vocab_size - 1,
                                                asr_transform=asr_transform,
                                                enc_type=enc_type,
                                                enc_proj=enc_proj,
                                                enc_kwargs=enc_kwargs)
        if dec_type != "xfmr":
            raise ValueError("XfmrTransducerASR: the decoder must be xfmr")
        att_dim = dec_kwargs["arch_kwargs"]["att_dim"]
        if not self.is_xfmr_encoder and enc_proj != att_dim:
            raise ValueError("enc_proj should be equal to att_dim")
        self.decoder = TorchTransformerDecoder(vocab_size, **dec_kwargs)

    def forward(self, x_pad: th.Tensor, x_len: NoneOrTensor, y_pad: th.Tensor,
                y_len: NoneOrTensor) -> AMForwardType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            y_len: N or None
        Return:
            enc_out: N x Ti x D
            dec_out: N x Ti x To+1 x V
            enc_len: N
        """
        # go through feature extractor & encoder
        enc_out, _, enc_len = self._training_prep(x_pad, x_len)
        # N x To+1
        tgt_pad = tf.pad(y_pad, (1, 0), value=self.blank)
        # N x Ti x To+1 x V
        dec_out = self.decoder(enc_out, tgt_pad, y_len + 1)
        return enc_out, dec_out, enc_len
