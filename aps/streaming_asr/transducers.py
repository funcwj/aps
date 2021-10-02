#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.streaming_asr.ctc import StreamingASREncoder
from aps.asr.beam_search.transducer import TransducerBeamSearch
from aps.asr.transducer.decoder import PyTorchRNNDecoder
from aps.asr.ctc import AMForwardType, NoneOrTensor
from aps.libs import ApsRegisters

from typing import Optional, Dict, List


@ApsRegisters.asr.register("streaming_asr@transducer")
class TransducerASR(StreamingASREncoder):
    """
    Transducer ASR with RNN decoders
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 40,
                 lctx: int = -1,
                 rctx: int = -1,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: int = -1,
                 dec_type: str = "rnn",
                 enc_kwargs: Dict = None,
                 dec_kwargs: Dict = None) -> None:
        super(TransducerASR, self).__init__(input_size,
                                            vocab_size,
                                            ctc=False,
                                            ead=True,
                                            lctx=lctx,
                                            rctx=rctx,
                                            asr_transform=asr_transform,
                                            enc_type=enc_type,
                                            enc_proj=enc_proj,
                                            enc_kwargs=enc_kwargs)
        self.blank = vocab_size - 1
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

    def greedy_search(self, x: th.Tensor) -> List[Dict]:
        """
        Greedy search for TransducerASR
        """
        beam_search_api = TransducerBeamSearch(self.decoder, blank=self.blank)
        with th.no_grad():
            enc_out = self._decoding_prep(x)
            return beam_search_api(enc_out, beam_size=1)

    def beam_search(self,
                    x: th.Tensor,
                    lm: Optional[nn.Module] = None,
                    lm_weight: float = 0,
                    beam_size: int = 16,
                    nbest: int = 8,
                    len_norm: bool = True,
                    **kwargs) -> List[Dict]:
        """
        Beam search for TransducerASR
        """
        beam_search_api = TransducerBeamSearch(self.decoder,
                                               lm=lm,
                                               blank=self.blank)
        with th.no_grad():
            enc_out = self._decoding_prep(x)
            return beam_search_api(enc_out,
                                   beam_size=beam_size,
                                   nbest=nbest,
                                   lm_weight=lm_weight,
                                   len_norm=len_norm)
