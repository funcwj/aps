#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from torch.nn.utils.rnn import pad_sequence

import aps.asr.beam_search.att as att_api
import aps.asr.beam_search.xfmr as xfmr_api

from typing import Optional, Dict, Tuple, List
from aps.asr.base.decoder import PyTorchRNNDecoder
from aps.asr.base.encoder import encoder_instance
from aps.asr.xfmr.encoder import support_xfmr_encoder
from aps.asr.xfmr.decoder import TorchTransformerDecoder
from aps.asr.base.attention import att_instance
from aps.libs import ApsRegisters

AttASROutputType = Tuple[th.Tensor, th.Tensor, Optional[th.Tensor],
                         Optional[th.Tensor]]
XfmrASROutputType = Tuple[th.Tensor, None, Optional[th.Tensor],
                          Optional[th.Tensor]]


class EncDecASRBase(nn.Module):
    """
    Base class for encoder/decoder ASR
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 30,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: Optional[int] = None,
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(EncDecASRBase, self).__init__()
        if eos < 0 or sos < 0:
            raise RuntimeError(f"Unsupported SOS/EOS value: {sos}/{eos}")
        self.sos = sos
        self.eos = eos
        self.asr_transform = asr_transform
        xfmr_encoder_cls = support_xfmr_encoder(enc_type)
        if xfmr_encoder_cls:
            self.encoder = xfmr_encoder_cls(input_size, **enc_kwargs)
            self.is_xfmr_encoder = True
            enc_proj = enc_kwargs["att_dim"]
        else:
            if enc_proj is None:
                raise ValueError(
                    "For non-transformer encoder, enc_proj can not be None")
            self.encoder = encoder_instance(enc_type, input_size, enc_proj,
                                            enc_kwargs)
            self.is_xfmr_encoder = False
        self.ctc = nn.Linear(enc_proj, vocab_size) if ctc else None

    def _batch_decoding_prep(self,
                             batch: List[th.Tensor],
                             max_len: int = -1) -> Tuple[th.Tensor]:
        """
        Prepare data for batch decoding
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
        max_len = max(lens) if max_len <= 0 else min(max(lens), max_len)
        # T x N x D
        enc_out = pad_sequence(outs, batch_first=False)
        enc_len = th.tensor(lens, device=enc_out.device)
        return enc_out, enc_len

    def _decoding_prep(self,
                       x: th.Tensor,
                       max_len: int = -1) -> Tuple[int, th.Tensor]:
        """
        Prepare data for decoding
        """
        # raw waveform
        if self.asr_transform:
            if x.dim() != 1:
                raise RuntimeError("Now only support for one utterance")
            # 1 x C x T x ... or 1 x T x F
            x, _ = self.asr_transform(x[None, ...], None)
        # already feature
        else:
            if x.dim() not in [2, 3]:
                raise RuntimeError(
                    f"Expect 2/3D(multi-channel) tensor, but got {x.dim()}")
            x = x[None, ...]
        # Ti x F
        inp_len = x.shape[-2]
        #  N x Ti x D
        enc_out, _ = self.encoder(x, None)
        # work out max_len
        max_len = inp_len if max_len <= 0 else min(inp_len, max_len)
        return max_len, enc_out

    def _training_prep(
        self, x_pad: th.Tensor, x_len: Optional[th.Tensor], y_pad: th.Tensor
    ) -> Tuple[th.Tensor, Optional[th.Tensor], th.Tensor, th.Tensor]:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
        Return:
            enc_out: N x Ti x D
            enc_len: N or None
            tgt_pad: N x To+1
        """
        # asr feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # N x Ti x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # N x To+1
        tgt_pad = tf.pad(y_pad, (1, 0), value=self.sos)
        # CTC branch
        enc_ctc = None
        if self.ctc:
            enc_ctc = self.ctc(enc_out)
        return enc_out, enc_len, enc_ctc, tgt_pad


@ApsRegisters.asr.register("att")
class AttASR(EncDecASRBase):
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
                 att_kwargs: Optional[Dict] = None,
                 enc_type: str = "common",
                 dec_type: str = "rnn",
                 enc_proj: Optional[int] = None,
                 enc_kwargs: Optional[Dict] = None,
                 dec_dim: int = 512,
                 dec_kwargs: Optional[Dict] = None) -> None:
        super(AttASR, self).__init__(input_size=input_size,
                                     vocab_size=vocab_size,
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
            enc_proj = enc_kwargs["att_dim"]
        self.att_net = att_instance(att_type, enc_proj, dec_dim, **att_kwargs)
        # TODO: make decoder flexible here
        self.decoder = PyTorchRNNDecoder(enc_proj,
                                         vocab_size - 1 if ctc else vocab_size,
                                         **dec_kwargs)

    def forward(self,
                x_pad: th.Tensor,
                x_len: Optional[th.Tensor],
                y_pad: th.Tensor,
                ssr: float = 0) -> AttASROutputType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
            ssr: schedule sampling rate
        Return:
            outs: N x (To+1) x V
            alis: N x (To+1) x T
        """
        # clear status
        self.att_net.clear()
        # go through feature extractor & encoder
        enc_out, enc_len, enc_ctc, tgt_pad = self._training_prep(
            x_pad, x_len, y_pad)
        # N x (To+1), pad SOS
        outs, alis = self.decoder(self.att_net,
                                  enc_out,
                                  enc_len,
                                  tgt_pad,
                                  schedule_sampling=ssr)
        return outs, alis, enc_ctc, enc_len

    def greedy_search(self,
                      x: th.Tensor,
                      max_len: int = -1,
                      len_norm: bool = True,
                      **kwargs) -> List[Dict]:
        """
        Greedy search (numbers should be same as beam_search with #beam-size == 1)
        Args
            x: audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            max_len, enc_out = self._decoding_prep(x, max_len=max_len)
            return att_api.greedy_search(self.decoder,
                                         self.att_net,
                                         enc_out,
                                         sos=self.sos,
                                         eos=self.eos,
                                         len_norm=len_norm)

    def beam_search(self,
                    x: th.Tensor,
                    max_len: int = -1,
                    **kwargs) -> List[Dict]:
        """
        Vectorized beam search
        Args
            x (Tensor): audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            max_len, enc_out = self._decoding_prep(x, max_len=max_len)
            return att_api.beam_search(self.decoder,
                                       self.att_net,
                                       enc_out,
                                       max_len=max_len,
                                       sos=self.sos,
                                       eos=self.eos,
                                       **kwargs)

    def beam_search_batch(self,
                          batch: List[th.Tensor],
                          max_len: int = -1,
                          **kwargs) -> List[Dict]:
        """
        Batch version of beam search
        Args
            batch (list[Tensor]): audio samples or acoustic features, S or Ti x F
        """
        with th.no_grad():
            enc_out, enc_len = self._batch_decoding_prep(batch, max_len=max_len)
            return att_api.beam_search_batch(self.decoder,
                                             self.att_net,
                                             enc_out,
                                             enc_len,
                                             max_len=max_len,
                                             sos=self.sos,
                                             eos=self.eos,
                                             **kwargs)


@ApsRegisters.asr.register("xfmr")
class XfmrASR(EncDecASRBase):
    """
    Attention based ASR model with (Non-)Transformer encoder + Transformer decoder
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 40,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "xfmr",
                 dec_type: str = "xfmr",
                 enc_proj: Optional[int] = None,
                 enc_kwargs: Optional[Dict] = None,
                 dec_kwargs: Optional[Dict] = None) -> None:
        super(XfmrASR, self).__init__(input_size=input_size,
                                      vocab_size=vocab_size,
                                      sos=sos,
                                      eos=eos,
                                      ctc=ctc,
                                      asr_transform=asr_transform,
                                      enc_type=enc_type,
                                      enc_proj=enc_proj,
                                      enc_kwargs=enc_kwargs)
        if dec_type != "xfmr":
            raise ValueError("XfmrASR: currently decoder must be xfmr")
        if not self.is_xfmr_encoder and enc_proj != dec_kwargs["att_dim"]:
            raise ValueError("enc_proj should be equal to att_dim")
        self.decoder = TorchTransformerDecoder(
            vocab_size - 1 if ctc else vocab_size, **dec_kwargs)

    def forward(self, x_pad: th.Tensor, x_len: Optional[th.Tensor],
                y_pad: th.Tensor) -> XfmrASROutputType:
        """
        Args:
            x_pad: N x Ti x D or N x S
            x_len: N or None
            y_pad: N x To
        Return:
            outs: N x (To+1) x V
        """
        # go through feature extractor & encoder
        enc_out, enc_len, enc_ctc, tgt_pad = self._training_prep(
            x_pad, x_len, y_pad)
        # N x To+1 x D
        dec_out = self.decoder(enc_out, enc_len, tgt_pad)
        return dec_out, None, enc_ctc, enc_len

    def beam_search(self,
                    x: th.Tensor,
                    beam: int = 16,
                    max_len: int = -1,
                    **kwargs) -> List[Dict]:
        """
        Beam search for Transformer
        """
        with th.no_grad():
            max_len, enc_out = self._decoding_prep(x, max_len=max_len)
            # beam search
            return xfmr_api.beam_search(self.decoder,
                                        enc_out,
                                        sos=self.sos,
                                        eos=self.eos,
                                        max_len=max_len,
                                        **kwargs)

    def beam_search_batch(self,
                          batch: List[th.Tensor],
                          max_len: int = -1,
                          **kwargs) -> List[Dict]:
        """
        Beam search for Transformer (batch version)
        """
        with th.no_grad():
            enc_out, enc_len = self._batch_decoding_prep(batch, max_len=max_len)
            # beam search
            return xfmr_api.beam_search_batch(self.decoder,
                                              enc_out,
                                              enc_len,
                                              sos=self.sos,
                                              eos=self.eos,
                                              max_len=max_len,
                                              **kwargs)
