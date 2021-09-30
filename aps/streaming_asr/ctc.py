#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.encoder import encoder_instance
from aps.streaming_asr.base.encoder import StreamingEncoder
from aps.asr.ctc import AMForwardType, NoneOrTensor
from aps.asr.beam_search.ctc import CtcApi
from aps.libs import ApsRegisters

from typing import Optional, Dict, List


class StreamingASREncoder(nn.Module):
    """
    Streaming ASR encoder class
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 ctc: bool = False,
                 ead: bool = False,
                 lctx: int = -1,
                 rctx: int = -1,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_proj: int = -1,
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(StreamingASREncoder, self).__init__()
        assert ctc or ead
        ctc_only = ctc and not ead
        # padding context of the network
        self.lctx = lctx
        self.rctx = rctx
        self.vocab_size = vocab_size
        self.asr_transform = asr_transform
        self.encoder = encoder_instance(enc_type, input_size,
                                        vocab_size if ctc_only else enc_proj,
                                        enc_kwargs, StreamingEncoder)
        # for hybrid ctc/aed, we add CTC branch
        self.ctc = nn.Linear(enc_proj, vocab_size) if ead and ctc else None

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
        # pad context
        if self.rctx + self.rctx > 0:
            x = tf.pad(x, (0, 0, self.rctx, self.rctx), "constant", 0)
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
        # pad context
        if self.lctx + self.rctx > 0:
            x_pad = tf.pad(x_pad, (0, 0, self.lctx, self.rctx), "constant", 0)
            x_len += self.lctx + self.rctx
        # N x Ti x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # CTC branch
        enc_ctc = enc_out
        if self.ctc:
            enc_ctc = self.ctc(enc_out)
        return enc_out, enc_ctc, enc_len


@ApsRegisters.asr.register("streaming_asr@ctc")
class CtcASR(StreamingASREncoder):
    """
    Streaming ASR (encoder + CTC)
    """

    def __init__(self,
                 input_size: int,
                 vocab_size: int,
                 ctc: bool = True,
                 ead: bool = False,
                 lctx: int = -1,
                 rctx: int = -1,
                 asr_transform: Optional[nn.Module] = None,
                 enc_type: str = "pytorch_rnn",
                 enc_kwargs: Optional[Dict] = None) -> None:
        super(CtcASR, self).__init__(input_size,
                                     vocab_size,
                                     ctc=ctc,
                                     ead=ead,
                                     lctx=lctx,
                                     rctx=rctx,
                                     asr_transform=asr_transform,
                                     enc_type=enc_type,
                                     enc_proj=-1,
                                     enc_kwargs=enc_kwargs)

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Make one processing step
        """
        return self.encoder.step(chunk)

    @th.jit.ignore
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
            # N x T x D or N x D x T
            enc_out = self._decoding_prep(x, batch_first=True)
            if self.ctc is not None:
                enc_out = self.ctc(enc_out)
            return ctc_api.beam_search(enc_out[0], **kwargs)
