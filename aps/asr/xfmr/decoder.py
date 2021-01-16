#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Union, Tuple, Optional
from aps.asr.xfmr.pose import get_xfmr_pose
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


class TransformerTorchDncoderLayer(TransformerDecoderLayer):
    """
    Wrapper for TransformerDecoderLayer (add pre-norm)
    """

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 pre_norm: bool = False,
                 dropout: bool = 0.1,
                 activation: str = "relu") -> None:
        super(TransformerTorchDncoderLayer,
              self).__init__(d_model,
                             nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation)
        self.pre_norm = pre_norm

    def ffn(self, src: th.Tensor) -> th.Tensor:
        """
        Get output of the feedforward network
        """
        return self.dropout3(
            self.linear2(self.dropout(self.activation(self.linear1(src)))))

    def forward(
            self,
            tgt: th.Tensor,
            memory: th.Tensor,
            tgt_mask: Optional[th.Tensor] = None,
            memory_mask: Optional[th.Tensor] = None,
            tgt_key_padding_mask: Optional[th.Tensor] = None,
            memory_key_padding_mask: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get decoder output (support pre_norm & post_norm)
        """
        inp = tgt
        if self.pre_norm:
            tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt,
                              tgt,
                              tgt,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = inp + self.dropout1(tgt2)
        if not self.pre_norm:
            tgt = self.norm1(tgt)

        inp = tgt
        if self.pre_norm:
            tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt,
                                   memory,
                                   memory,
                                   attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = inp + self.dropout2(tgt2)
        if not self.pre_norm:
            tgt = self.norm2(tgt)

        inp = tgt
        if self.pre_norm:
            tgt = self.norm3(tgt)
        tgt = inp + self.ffn(tgt2)
        if not self.pre_norm:
            tgt = self.norm3(tgt)
        return tgt


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
                 num_layers: int = 6,
                 post_norm: bool = True) -> None:
        super(TorchTransformerDecoder, self).__init__()
        # default normal init (std=1), do not need to scale
        self.vocab_embed = nn.Embedding(vocab_size, att_dim)
        # use absolute positional embedding here
        self.abs_pos_enc = get_xfmr_pose("inp_sin",
                                         att_dim,
                                         dropout=pos_dropout,
                                         scale_embed=scale_embed)
        decoder_layer = TransformerTorchDncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = nn.LayerNorm(att_dim) if not post_norm else None
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_layers,
                                          norm=final_norm)
        self.output = nn.Linear(att_dim, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def step(self,
             enc_out: th.Tensor,
             tgt_pad: th.Tensor,
             enc_len: Optional[th.Tensor] = None,
             pre_emb: Optional[th.Tensor] = None,
             out_idx: Optional[int] = None) -> Tuple[th.Tensor]:
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
        # N x T x E
        tgt_emb = self.vocab_embed(tgt_pad)
        # T x N x E
        tgt_emb = self.abs_pos_enc(tgt_emb, t=offset)
        # T+T' x N x E
        if pre_emb is not None:
            tgt_emb = th.cat([pre_emb, tgt_emb], dim=0)
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
            enc_out (Tensor): N x T x D
            enc_len (Tensor): N or None
            tgt_pad (Tensor): N x To
        Return:
            dec_out (Tensor): N x T x D
        """
        # T x N x V
        dec_out, _ = self.step(enc_out.transpose(0, 1),
                               tgt_pad,
                               enc_len=enc_len)
        return dec_out.transpose(0, 1)
