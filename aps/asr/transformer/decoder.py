#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Union, Tuple, Optional
from aps.asr.transformer.pose import InputSinPosEncoding
from aps.asr.base.attention import padding_mask


def prep_sub_mask(T: int, device: Union[str, th.device] = "cpu") -> th.Tensor:
    """
    Prepare a square sub-sequence masks (-inf/0)
    egs: for N = 8, output
    tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., -inf, -inf, -inf, -inf],
        [0., 0., 0., 0., 0., -inf, -inf, -inf],
        [0., 0., 0., 0., 0., 0., -inf, -inf],
        [0., 0., 0., 0., 0., 0., 0., -inf],
        [0., 0., 0., 0., 0., 0., 0., 0.]])
    """
    mask = (th.triu(th.ones(T, T, device=device), diagonal=1) == 1).float()
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


class TorchTransformerDecoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 vocab_size: int,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0,
                 att_dropout: float = 0.1,
                 pos_enc: bool = True,
                 num_layers: int = 6) -> None:
        super(TorchTransformerDecoder, self).__init__()
        self.vocab_embed = nn.Embedding(vocab_size, att_dim)
        self.abs_pos_enc = InputSinPosEncoding(att_dim,
                                               dropout=pos_dropout,
                                               scale_embed=scale_embed)
        decoder_layer = TransformerDecoderLayer(att_dim,
                                                nhead,
                                                dim_feedforward=feedforward_dim,
                                                dropout=att_dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(att_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def step(self,
             enc_out: th.Tensor,
             tgt_pad: th.Tensor,
             enc_len: Optional[th.Tensor] = None,
             pre_emb: Optional[th.Tensor] = None,
             out_idx: Optional[int] = None,
             point: Optional[th.Tensor] = None) -> Tuple[th.Tensor]:
        """
        Args:
            enc_out (Tensor): T x N x D
            tgt_pad (Tensor): N x To
            enc_len (Tensor): N or None
            pre_emb (Tensor): T' x N x D
        Return:
            dec_out (Tensor): T+T' x N x D or N x D
            tgt_emb (Tensor): T+T' x N x E
        """
        # N x Ti
        offset = 0 if pre_emb is None else pre_emb.shape[0]
        memory_mask = None if enc_len is None else (padding_mask(enc_len) == 1)
        tgt_mask = prep_sub_mask(tgt_pad.shape[-1] + offset,
                                 device=tgt_pad.device)
        tgt_emb = self.vocab_embed(tgt_pad)
        if offset:
            # T + T' x N x E
            tgt_emb = self.abs_pos_enc(tgt_emb, t=offset)
            if point is not None:
                pre_emb = pre_emb[:, point]
            tgt_emb = th.cat([pre_emb, tgt_emb], dim=0)
        else:
            # T x N x E
            tgt_emb = self.abs_pos_enc(tgt_emb)
        # To+1 x N x D
        dec_out = self.decoder(tgt_emb,
                               enc_out,
                               tgt_mask=tgt_mask,
                               memory_mask=None,
                               tgt_key_padding_mask=None,
                               memory_key_padding_mask=memory_mask)
        if out_idx is not None:
            dec_out = dec_out[out_idx]
        # To+1 x N x V
        dec_out = self.output(dec_out)
        return dec_out, tgt_emb

    def forward(self, enc_out: th.Tensor, enc_len: Optional[th.Tensor],
                tgt_pad: th.Tensor) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): T x N x D
            enc_len (Tensor): N or None
            tgt_pad (Tensor): N x To
        Return:
            dec_out (Tensor): T x N x D
        """
        # T x N x V
        dec_out, _ = self.step(enc_out, tgt_pad, enc_len=enc_len)
        return dec_out
