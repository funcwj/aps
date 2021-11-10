#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

import aps.asr.beam_search.att as att_api
import aps.asr.beam_search.transformer as xfmr_api

from typing import Optional, Dict, List
from aps.asr.ctc import CtcASR, NoneOrTensor, AMForwardType
from aps.asr.base.decoder import TorchRNNDecoder
from aps.asr.transformer.decoder import TorchTransformerDecoder
from aps.asr.base.attention import att_instance
from aps.asr.beam_search.ctc import CtcApi
from aps.libs import ApsRegisters


class ASREncoderDecoderBase(CtcASR):
    """
    Base class for encoder/decoder attention based AM
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: int = -1,
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(ASREncoderDecoderBase, self).__init__(input_size,
                                                    vocab_size,
                                                    ctc=ctc,
                                                    ead=True,
                                                    asr_transform=asr_transform,
                                                    enc_type=enc_type,
                                                    enc_proj=enc_proj,
                                                    enc_kwargs=enc_kwargs)
        if eos < 0 or sos < 0:
            raise RuntimeError(f"Unsupported SOS/EOS value: {sos}/{eos}")
        self.sos = sos
        self.eos = eos


@ApsRegisters.asr.register("asr@att")
class AttASR(ASREncoderDecoderBase):
    """
    Attention based ASR model with (Non-)Transformer encoder + attention + RNN decoder
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 30,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 att_type: str = "ctx",
                 att_kwargs: Dict = {},
                 enc_type: str = "common",
                 dec_type: str = "rnn",
                 enc_proj: int = -1,
                 enc_kwargs: Dict = {},
                 dec_dim: int = 512,
                 dec_kwargs: Dict = {}) -> None:
        super(AttASR, self).__init__(input_size,
                                     vocab_size,
                                     sos=sos,
                                     eos=eos,
                                     ctc=ctc,
                                     asr_transform=asr_transform,
                                     enc_type=enc_type,
                                     enc_proj=enc_proj,
                                     enc_kwargs=enc_kwargs)
        if dec_type != "rnn":
            raise ValueError("AttASR: currently decoder must be rnn")
        if self.is_xfmr_encoder:
            enc_proj = enc_kwargs["arch_kwargs"]["att_dim"]
        self.att_net = att_instance(att_type, enc_proj, dec_dim, **att_kwargs)
        # TODO: make decoder flexible here
        self.decoder = TorchRNNDecoder(enc_proj,
                                       vocab_size - 1 if ctc else vocab_size,
                                       **dec_kwargs)

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
        # clear status
        self.att_net.clear()
        # go through feature extractor & encoder
        enc_out, enc_ctc, enc_len = self._training_prep(x_pad, x_len)
        # N x To+1
        tgt_pad = tf.pad(y_pad, (1, 0), value=self.sos)
        # N x (To+1), pad SOS
        dec_out, _ = self.decoder(self.att_net,
                                  enc_out,
                                  enc_len,
                                  tgt_pad,
                                  schedule_sampling=ssr)
        return dec_out, enc_ctc, enc_len

    def greedy_search(self,
                      x: th.Tensor,
                      len_norm: bool = True,
                      **kwargs) -> List[Dict]:
        """
        Greedy search (numbers should be same as beam_search with #beam-size == 1)
        Args
            x: audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            enc_out = self._decoding_prep(x)
            return att_api.greedy_search(self.decoder,
                                         self.att_net,
                                         enc_out,
                                         sos=self.sos,
                                         eos=self.eos,
                                         len_norm=len_norm)

    def ctc_att_rescore(self,
                        x: th.Tensor,
                        ctc_weight: float = 0,
                        len_norm: bool = False,
                        **kwargs) -> List[Dict]:
        """
        Decoder rescore for CTC nbest results
        Args
            x: audio samples or acoustic features, S or Ti x F
        """
        ctc_api = CtcApi(self.vocab_size - 1)
        with th.no_grad():
            # N x T x D
            enc_out = self._decoding_prep(x)
            if self.ctc is None:
                raise RuntimeError(
                    "Can't do CTC beam search as self.ctc is None")
            ctc_nbest = ctc_api.beam_search(self.ctc(enc_out)[0],
                                            sos=self.sos,
                                            eos=self.eos,
                                            len_norm=False,
                                            **kwargs)
            return att_api.decoder_rescore(ctc_nbest,
                                           self.decoder,
                                           self.att_net,
                                           enc_out,
                                           ctc_weight=ctc_weight,
                                           len_norm=len_norm)

    def beam_search(self,
                    x: th.Tensor,
                    ctc_weight: float = 0,
                    **kwargs) -> List[Dict]:
        """
        Vectorized beam search
        Args
            x (Tensor): audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            # N x T x D
            enc_out = self._decoding_prep(x)
            ctc_prob = self.ctc(enc_out)[0] if self.ctc else None
            if ctc_weight < 1:
                return att_api.beam_search(self.decoder,
                                           self.att_net,
                                           enc_out,
                                           ctc_prob=ctc_prob,
                                           ctc_weight=ctc_weight,
                                           sos=self.sos,
                                           eos=self.eos,
                                           **kwargs)
            else:
                if ctc_prob is None:
                    raise RuntimeError(
                        "Can't do CTC beam search as self.ctc is None")
                ctc_api = CtcApi(self.vocab_size - 1)
                return ctc_api.beam_search(ctc_prob,
                                           sos=self.sos,
                                           eos=self.eos,
                                           **kwargs)

    def beam_search_batch(self, batch: List[th.Tensor], **kwargs) -> List[Dict]:
        """
        Batch version of beam search
        Args
            batch (list[Tensor]): audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            enc_out, enc_len = self._batch_decoding_prep(batch)
            return att_api.beam_search_batch(self.decoder,
                                             self.att_net,
                                             enc_out,
                                             enc_len,
                                             sos=self.sos,
                                             eos=self.eos,
                                             **kwargs)


@ApsRegisters.asr.register("asr@xfmr")
class XfmrASR(ASREncoderDecoderBase):
    """
    Attention based AM with (Non-)Transformer encoder + Transformer decoder
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "xfmr",
                 dec_type: str = "xfmr",
                 enc_proj: int = -1,
                 enc_kwargs: Dict = {},
                 dec_kwargs: Dict = {}) -> None:
        super(XfmrASR, self).__init__(input_size,
                                      vocab_size,
                                      sos=sos,
                                      eos=eos,
                                      ctc=ctc,
                                      asr_transform=asr_transform,
                                      enc_type=enc_type,
                                      enc_proj=enc_proj,
                                      enc_kwargs=enc_kwargs)
        if dec_type != "xfmr":
            raise ValueError("XfmrASR: currently decoder must be xfmr")
        att_dim = dec_kwargs["arch_kwargs"]["att_dim"]
        if not self.is_xfmr_encoder and enc_proj != att_dim:
            raise ValueError(
                f"enc_proj({enc_proj}) should be equal to att_dim({att_dim})")
        self.decoder = TorchTransformerDecoder(
            vocab_size - 1 if ctc else vocab_size, **dec_kwargs)

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
            y_len: N or None
            ssr: not used here, left for future
        Return:
            dec_out: N x (To+1) x V
            enc_ctc: N x T x V or None
            enc_len: N or None
        """
        # go through feature extractor & encoder
        enc_out, enc_ctc, enc_len = self._training_prep(x_pad, x_len)
        # N x To+1
        tgt_pad = tf.pad(y_pad, (1, 0), value=self.sos)
        # N x To+1 x D
        dec_out = self.decoder(enc_out, enc_len, tgt_pad, y_len + 1)
        return dec_out, enc_ctc, enc_len

    def greedy_search(self,
                      x: th.Tensor,
                      len_norm: bool = True,
                      **kwargs) -> List[Dict]:
        """
        Greedy search (numbers should be same as beam_search with #beam-size == 1)
        Args
            x: audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            enc_out = self._decoding_prep(x, batch_first=False)
            return xfmr_api.greedy_search(self.decoder,
                                          enc_out,
                                          sos=self.sos,
                                          eos=self.eos,
                                          len_norm=len_norm)

    def ctc_att_rescore(self,
                        x: th.Tensor,
                        ctc_weight: float = 0,
                        len_norm: bool = False,
                        **kwargs) -> List[Dict]:
        """
        Decoder rescore for CTC nbest results
        Args
            x: audio samples or acoustic features, S or Ti x F
        """
        ctc_api = CtcApi(self.vocab_size - 1)
        with th.no_grad():
            # T x N x D
            enc_out = self._decoding_prep(x, batch_first=False)
            if self.ctc is None:
                raise RuntimeError(
                    "Can't do CTC beam search as self.ctc is None")
            ctc_nbest = ctc_api.beam_search(self.ctc(enc_out)[:, 0],
                                            sos=self.sos,
                                            eos=self.eos,
                                            len_norm=False,
                                            **kwargs)
            return xfmr_api.decoder_rescore(ctc_nbest,
                                            self.decoder,
                                            enc_out,
                                            ctc_weight=ctc_weight,
                                            len_norm=len_norm)

    def beam_search(self,
                    x: th.Tensor,
                    ctc_weight: float = 0,
                    **kwargs) -> List[Dict]:
        """
        Beam search for Transformer
        """
        with th.no_grad():
            # T x N x D
            enc_out = self._decoding_prep(x, batch_first=False)
            ctc_prob = self.ctc(enc_out)[:, 0] if self.ctc else None
            # beam search
            if ctc_weight < 1:
                return xfmr_api.beam_search(self.decoder,
                                            enc_out,
                                            ctc_prob=ctc_prob,
                                            ctc_weight=ctc_weight,
                                            sos=self.sos,
                                            eos=self.eos,
                                            **kwargs)
            else:
                if ctc_prob is None:
                    raise RuntimeError(
                        "Can't do CTC beam search as self.ctc is None")
                ctc_api = CtcApi(self.vocab_size - 1)
                return ctc_api.beam_search(ctc_prob,
                                           sos=self.sos,
                                           eos=self.eos,
                                           **kwargs)

    def beam_search_batch(self, batch: List[th.Tensor], **kwargs) -> List[Dict]:
        """
        Beam search for Transformer (batch version)
        """
        with th.no_grad():
            enc_out, enc_len = self._batch_decoding_prep(batch,
                                                         batch_first=False)
            # beam search
            return xfmr_api.beam_search_batch(self.decoder,
                                              enc_out,
                                              enc_len,
                                              sos=self.sos,
                                              eos=self.eos,
                                              **kwargs)
