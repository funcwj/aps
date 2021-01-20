#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Dict
from aps.asr.base.attention import padding_mask
from aps.asr.base.encoder import EncRetType
from aps.asr.xfmr.impl import get_xfmr_encoder
from aps.asr.xfmr.pose import get_xfmr_pose
from aps.asr.xfmr.proj import get_xfmr_proj


class TransformerEncoder(nn.Module):
    """
    Transformer based encoders. Currently it supports {xfmr|cfmr}_{abs|rel|xl}
    The relationship between type of the encoder and positional encoding layer:
        {xfmr|cfmr}_abs <=> inp_sin
        {xfmr|cfmr}_rel <=> rel
        {xfmr|cfmr}_xl  <=> sin
    """

    def __init__(self,
                 enc_type: str,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_kwargs: Optional[Dict] = None,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 num_layers: int = 6,
                 radius: int = 128,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 kernel_size: int = 16,
                 post_norm: bool = True,
                 untie_rel: bool = True):
        super(TransformerEncoder, self).__init__()
        self.type = enc_type.split("_")[-1]
        self.proj = get_xfmr_proj(proj_layer, input_size, att_dim, proj_kwargs)
        self.pose = get_xfmr_pose(enc_type,
                                  att_dim,
                                  nhead=nhead,
                                  radius=radius,
                                  dropout=pos_dropout,
                                  scale_embed=scale_embed)
        self.encoder = get_xfmr_encoder(enc_type,
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        att_dropout=att_dropout,
                                        ffn_dropout=ffn_dropout,
                                        kernel_size=kernel_size,
                                        pre_norm=not post_norm,
                                        untie_rel=untie_rel)

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Go through projection layer
        Args:
            inp_pad: N x Ti x F
            inp_len: N or None
        Return:
            enc_inp: N x Ti x D
            inp_len: N or None
            src_pad_mask: N x Ti or None
        """
        inp_len = self.proj.num_frames(inp_len)
        enc_inp = self.proj(inp_pad)
        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)

        if self.type == "abs":
            # enc_inp: N x Ti x D => Ti x N x D
            enc_inp = self.pose(enc_inp)
            inj_pose = None
        else:
            # enc_inp: N x Ti x D => Ti x N x D
            enc_inp = enc_inp.transpose(0, 1)
            nframes = enc_inp.shape[0]
            # 2Ti-1 x D
            if self.type == "rel":
                inj_pose = self.pose(
                    th.arange(-nframes + 1, nframes, device=enc_inp.device))
            else:
                inj_pose = self.pose(
                    th.arange(0, 2 * nframes - 1, 1.0, device=enc_inp.device))
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
                               inj_pose=inj_pose,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len
