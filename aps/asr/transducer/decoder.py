#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Tuple, Dict
from aps.asr.transformer.impl import get_xfmr_encoder
from aps.asr.transformer.pose import get_xfmr_pose
from aps.asr.transformer.utils import prep_sub_mask
from aps.asr.base.attention import padding_mask
from aps.asr.base.decoder import LayerNormRNN
from aps.asr.base.component import OneHotEmbedding, PyTorchRNN


class DecoderBase(nn.Module):
    """
    Base class for RNNT decoders
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512,
                 enc_dim: int = 512,
                 dec_dim: int = 512,
                 jot_dim: int = 512,
                 onehot_embed: bool = False) -> None:
        super(DecoderBase, self).__init__()
        if not onehot_embed:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        self.enc_proj = nn.Linear(enc_dim, jot_dim, bias=False)
        self.dec_proj = nn.Linear(dec_dim, jot_dim)
        self.vocab_size = vocab_size
        self.output = nn.Linear(jot_dim, vocab_size, bias=False)

    def joint(self, enc_proj_out: th.Tensor,
              dec_proj_out: th.Tensor) -> th.Tensor:
        """
        Joint network prediction (without projection)
        Args:
            enc_proj_out: N x Ti x J or N x J
            dec_proj_out: N x To+1 x J or N x J
        Return:
            output: N x Ti x To+1 x V or N x 1 x V
        """
        # N x Ti x To+1 x J or N x 1 x J
        add_out = enc_proj_out.unsqueeze(-2) + dec_proj_out.unsqueeze(1)
        # N x Ti x To+1 x V or N x 1 x V
        return self.output(th.tanh(add_out))


class PyTorchRNNDecoder(DecoderBase):
    """
    Wrapper for pytorch's RNN Decoder
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512,
                 enc_dim: int = 512,
                 jot_dim: int = 512,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 proj_size: int = -1,
                 add_ln: bool = False,
                 dropout: float = 0.0,
                 onehot_embed: bool = False) -> None:
        super(PyTorchRNNDecoder, self).__init__(vocab_size,
                                                embed_size=embed_size,
                                                enc_dim=enc_dim,
                                                dec_dim=hidden,
                                                jot_dim=jot_dim,
                                                onehot_embed=onehot_embed)
        # uni-dir RNNs
        if add_ln:
            self.decoder = LayerNormRNN(rnn,
                                        embed_size,
                                        hidden,
                                        proj_size=proj_size,
                                        num_layers=num_layers,
                                        dropout=dropout,
                                        bidirectional=False)
        else:
            self.decoder = PyTorchRNN(rnn,
                                      embed_size,
                                      hidden,
                                      proj_size=proj_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      bidirectional=False)

    def forward(self, enc_out: th.Tensor, tgt_pad: th.Tensor) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): N x Ti x D
            tgt_pad (Tensor): N x To+1 (padding blank at time = 0)
        Return:
            output: N x Ti x To+1 x V
        """
        # N x To+1 x E
        tgt_pad = self.vocab_embed(tgt_pad)
        # N x To+1 x D
        dec_out = self.decoder(tgt_pad)[0]
        # N x Ti x J
        enc_out = self.enc_proj(enc_out)
        # N x To+1 x J
        dec_out = self.dec_proj(dec_out)
        # N x Ti x To+1 x V
        return self.joint(enc_out, dec_out)

    def pred(self, pred_prev, hidden=None):
        """
        Make one prediction step for decoder
        Args:
            pred_prev: N x 1
        Return:
            dec_out: N x J
        """
        pred_prev_emb = self.vocab_embed(pred_prev)  # N x 1 x E
        dec_out, hidden = self.decoder(pred_prev_emb, hidden)
        dec_out = self.dec_proj(dec_out[:, -1])
        return dec_out, hidden


class TorchTransformerDecoder(DecoderBase):
    """
    Vanilla Transformer encoder as transducer decoder
    """

    def __init__(self,
                 vocab_size: int,
                 enc_dim: Optional[int] = None,
                 jot_dim: int = 512,
                 att_dim: int = 512,
                 pose_kwargs: Dict = {},
                 arch_kwargs: Dict = {},
                 num_layers: int = 6,
                 onehot_embed: bool = False) -> None:
        super(TorchTransformerDecoder,
              self).__init__(vocab_size,
                             enc_dim=enc_dim if enc_dim else att_dim,
                             dec_dim=att_dim,
                             jot_dim=jot_dim,
                             onehot_embed=onehot_embed)
        self.abs_pos_enc = get_xfmr_pose("abs", att_dim, **pose_kwargs)
        self.decoder = get_xfmr_encoder("xfmr", "abs", num_layers, arch_kwargs)

    def forward(self, enc_out: th.Tensor, tgt_pad: th.Tensor,
                tgt_len: Optional[th.Tensor]) -> th.Tensor:
        """
        Args:
            enc_out (Tensor): N x Ti x D
            tgt_pad (Tensor): N x To+1 (padding blank at time = 1)
            tgt_len (Tensor): N or None
        Return:
            output: N x Ti x To+1 x V
        """
        # N x Ti
        pad_mask = None if tgt_len is None else (padding_mask(tgt_len) == 1)
        # genrarte target masks (-inf/0)
        tgt_mask = prep_sub_mask(tgt_pad.shape[-1], device=tgt_pad.device)
        # To+1 x N x E
        tgt_pad = self.abs_pos_enc(self.vocab_embed(tgt_pad))
        # To+1 x N x D
        dec_out = self.decoder(tgt_pad,
                               src_mask=tgt_mask,
                               src_key_padding_mask=pad_mask)
        # N x Ti x J
        enc_out = self.enc_proj(enc_out)
        # N x To+1 x J
        dec_out = self.dec_proj(dec_out.transpose(0, 1))
        return self.joint(enc_out, dec_out)

    def pred(self,
             pred_prev: th.Tensor,
             hidden: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        Make one prediction step for decoder
        Args:
            pred_prev: N x 1
            hidden: None or T x N x E
        Return:
            dec_out: N x J
        """
        t = 0 if hidden is None else hidden.shape[0]
        # 1 x 1 x E
        pred_prev_emb = self.abs_pos_enc(self.vocab_embed(pred_prev), t=t)
        hidden = pred_prev_emb if hidden is None else th.cat(
            [hidden, pred_prev_emb], dim=0)
        tgt_mask = prep_sub_mask(t + 1, device=pred_prev.device)
        dec_out = self.decoder(hidden, mask=tgt_mask)
        dec_out = self.dec_proj(dec_out[-1])
        return dec_out, hidden
