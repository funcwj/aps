#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict, Tuple, Union, List

from aps.asr.transformer.decoder import TorchTransformerDecoder
from aps.asr.transformer.encoder import support_xfmr_encoder
from aps.asr.base.encoder import encoder_instance


class TransformerASR(nn.Module):
    """
    Transformer-based end-to-end ASR
    """

    def __init__(self,
                 input_size: int = 80,
                 vocab_size: int = 40,
                 sos: int = -1,
                 eos: int = -1,
                 ctc: bool = False,
                 asr_transform: Optional[nn.Module] = None,
                 encoder_type: str = "transformer",
                 encoder_proj: Optional[int] = None,
                 encoder_kwargs: Optional[Dict] = None,
                 decoder_type: str = "transformer",
                 decoder_kwargs: Optional[Dict] = None) -> None:
        super(TransformerASR, self).__init__()
        if eos < 0 or sos < 0:
            raise RuntimeError(f"Unsupported SOS/EOS value: {sos}/{eos}")
        xfmr_encoder_cls = support_xfmr_encoder(encoder_type)
        if xfmr_encoder_cls:
            self.encoder = xfmr_encoder_cls(input_size, **encoder_kwargs)
        else:
            if encoder_proj is None:
                raise ValueError("For non-transformer encoder, "
                                 "encoder_proj can not be None")
            self.encoder = encoder_instance(encoder_type, input_size,
                                            encoder_proj, **encoder_kwargs)
        if decoder_type != "transformer":
            raise ValueError("TransformerASR: decoder must be transformer")
        self.decoder = TorchTransformerDecoder(vocab_size -
                                               1 if ctc else vocab_size,
                                               enc_dim=encoder_proj,
                                               **decoder_kwargs)
        self.sos = sos
        self.eos = eos
        self.asr_transform = asr_transform
        self.ctc = nn.Linear(decoder_kwargs["att_dim"],
                             vocab_size) if ctc else None

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
        Return:
            outs: N x (To+1) x V
        """
        # feature transform
        if self.asr_transform:
            x_pad, x_len = self.asr_transform(x_pad, x_len)
        # Ti x N x D
        enc_out, enc_len = self.encoder(x_pad, x_len)
        # CTC
        if self.ctc:
            ctc_branch = self.ctc(enc_out)
            ctc_branch = ctc_branch.transpose(0, 1)
        else:
            ctc_branch = None
        # To+1 x N x D
        dec_out = self.decoder(enc_out, enc_len, y_pad, sos=self.sos)
        # N x To+1 x D
        dec_out = dec_out.transpose(0, 1).contiguous()
        return dec_out, None, ctc_branch, enc_len

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
        Beam search for Transformer
        """
        with th.no_grad():
            # raw wave
            if self.asr_transform:
                if x.dim() != 1:
                    raise RuntimeError("Now only support for one utterance")
                x, _ = self.asr_transform(x[None, ...], None)
            else:
                # T x F or Beam x T x F
                if x.dim() not in [2, 3]:
                    raise RuntimeError(f"Expect 2/3D tensor, but got {x.dim()}")
                x = x[None, ...]
            # Ti x N x D
            enc_out, _ = self.encoder(x, None)
            # beam search
            return self.decoder.beam_search(enc_out,
                                            beam=beam,
                                            lm=lm,
                                            lm_weight=lm_weight,
                                            sos=self.sos,
                                            eos=self.eos,
                                            nbest=nbest,
                                            max_len=max_len,
                                            vectorized=vectorized,
                                            normalized=normalized)
