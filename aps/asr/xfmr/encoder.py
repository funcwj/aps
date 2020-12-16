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
from aps.asr.xfmr.utils import get_xfmr_proj
from aps.libs import Register

TransformerEncoders = Register("xfmr_encoder")


def support_xfmr_encoder(encoder_name: str) -> Optional[nn.Module]:
    """
    Return transformer decoder
    """
    if encoder_name in TransformerEncoders:
        return TransformerEncoders[encoder_name]
    else:
        return None


@TransformerEncoders.register("xfmr")
class TorchTransformerEncoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 post_norm: bool = True,
                 pos_enc: bool = True,
                 num_layers: int = 6) -> None:
        super(TorchTransformerEncoder, self).__init__()
        self.abs_pos_enc = get_xfmr_pose("inp_sin",
                                         att_dim,
                                         dropout=pos_dropout,
                                         scale_embed=scale_embed)
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)
        self.encoder = get_xfmr_encoder("xfmr",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        dropout=att_dropout,
                                        pre_norm=not post_norm)

    def forward(self, x_pad: th.Tensor,
                x_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        x_len = self.src_proj.num_frames(x_len)
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.abs_pos_enc(self.src_proj(x_pad))
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


@TransformerEncoders.register("xfmr_rel")
class RelTransformerEncoder(nn.Module):
    """
    Using relative position encoding
    """

    def __init__(self,
                 input_size,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 k_dim: int = 128,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 post_norm: bool = True,
                 value_rel_pose: bool = False,
                 num_layers: int = 6) -> None:
        super(RelTransformerEncoder, self).__init__()
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)
        embed_dim = att_dim // nhead
        self.key_rel_pose = get_xfmr_pose("rel",
                                          embed_dim,
                                          radius=k_dim,
                                          dropout=pos_dropout)
        if value_rel_pose:
            self.val_rel_pose = get_xfmr_pose("rel",
                                              embed_dim,
                                              radius=k_dim,
                                              dropout=pos_dropout)
        else:
            self.val_rel_pose = None

        self.encoder = get_xfmr_encoder("xfmr_rel",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        dropout=att_dropout,
                                        pre_norm=not post_norm)

    def _get_relative_embed(self, inp: th.Tensor) -> EncRetType:
        """
        Return relative embeddings
        """
        key_rel_enc = self.key_rel_pose(inp.shape[0])
        val_rel_enc = self.val_rel_pose(
            inp.shape[0]) if self.val_rel_pose else None
        return key_rel_enc, val_rel_enc

    def forward(self, x_pad: th.Tensor,
                x_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.src_proj(x_pad).transpose(0, 1)
        x_len = self.src_proj.num_frames(x_len)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # rel encodings
        key_rel_enc, value_rel_enc = self._get_relative_embed(x_emb)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               src_mask=None,
                               key_rel_pose=key_rel_enc,
                               value_rel_pose=value_rel_enc,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


@TransformerEncoders.register("xfmr_xl")
class RelXLTransformerEncoder(nn.Module):
    """
    Using relative position encoding in Transformer-XL
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 post_norm: bool = True,
                 untie_rel: bool = True,
                 num_layers: int = 6) -> None:
        super(RelXLTransformerEncoder, self).__init__()
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)
        self.sin_pose = get_xfmr_pose("sin", att_dim, dropout=pos_dropout)
        self.encoder = get_xfmr_encoder("xfmr_xl",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        dropout=att_dropout,
                                        pre_norm=not post_norm,
                                        untie_rel=untie_rel)

    def forward(self, x_pad: th.Tensor,
                x_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.src_proj(x_pad).transpose(0, 1)
        x_len = self.src_proj.num_frames(x_len)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # Ti x D
        sin_pos_enc = self.sin_pose(x_emb.shape[0])
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               sin_pose=sin_pos_enc,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


@TransformerEncoders.register("conformer")
class ConformerEncoder(nn.Module):
    """
    Conformer encoder
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 kernel_size: int = 16,
                 untie_rel: bool = True,
                 num_layers: int = 6) -> None:
        super(ConformerEncoder, self).__init__()
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)
        self.sin_pose = get_xfmr_pose("sin", att_dim, dropout=pos_dropout)
        self.encoder = get_xfmr_encoder("conformer",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        dropout=att_dropout,
                                        kernel_size=kernel_size,
                                        pre_norm=False,
                                        untie_rel=untie_rel)

    def forward(self, x_pad: th.Tensor,
                x_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.src_proj(x_pad).transpose(0, 1)
        x_len = self.src_proj.num_frames(x_len)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # Ti x D
        sin_pos_enc = self.sin_pose(x_emb.shape[0])
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               sin_pose=sin_pos_enc,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len
