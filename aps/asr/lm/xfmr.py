# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Tuple
from aps.asr.xfmr.pose import get_xfmr_pose
from aps.asr.xfmr.impl import get_xfmr_encoder
from aps.asr.xfmr.decoder import prep_sub_mask
from aps.asr.base.attention import padding_mask
from aps.libs import ApsRegisters


@ApsRegisters.asr.register("asr@xfmr_lm")
class TorchXfmrLM(nn.Module):
    """
    Torch Transformer LM
    """

    def __init__(self,
                 vocab_size: int = 40,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 num_layers: int = 6) -> None:
        super(TorchXfmrLM, self).__init__()
        self.vocab_embed = nn.Embedding(vocab_size, att_dim)
        self.abs_pos_enc = get_xfmr_pose("xfmr_abs",
                                         att_dim,
                                         dropout=pos_dropout,
                                         scale_embed=scale_embed)
        self.encoder = get_xfmr_encoder("xfmr_abs",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        att_dropout=att_dropout,
                                        ffn_dropout=ffn_dropout)
        # output distribution
        self.dist = nn.Linear(att_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(
            self,
            token: th.Tensor,
            hidden: Optional[th.Tensor] = None,
            token_len: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        args:
            token: input token sequence, N x T
            hidden: previous sequence embeddings, T x N x E
            token_len: length of x, N or None
        return:
            output: N x T x V
            hidden: current sequence embeddings, T x N x E
        """
        # N x T => T x N x V
        t = 0 if hidden is None else hidden.shape[0]
        token_embed = self.abs_pos_enc(self.vocab_embed(token), t=t)
        # h == None: training or eval in time = 0
        hidden = token_embed if hidden is None else th.cat(
            [hidden, token_embed], dim=0)
        # tgt_mask: T x T
        tgt_mask = prep_sub_mask(hidden.shape[0], device=hidden.device)
        # src_pad_mask: N x T
        src_pad_mask = None if token_len is None else (padding_mask(token_len)
                                                       == 1)
        # Ti x N x D
        enc_out = self.encoder(hidden,
                               inj_pose=None,
                               src_mask=tgt_mask,
                               src_key_padding_mask=src_pad_mask)
        # Ti x N x V
        output = self.dist(enc_out)
        # N x Ti x V
        return output.transpose(0, 1), hidden
