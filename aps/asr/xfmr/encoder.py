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

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp_pad: N x Ti x F
            inp_len: N or None
        Return:
            enc_out: N x Ti x D (keep same as aps.asr.base.encoder)
        """
        inp_len = self.src_proj.num_frames(inp_len)
        # inp_sub: N x Ti x D => Ti x N x D
        inp_sub = self.abs_pos_enc(self.src_proj(inp_pad))
        # src_pad_mask: N x Ti
        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        # Ti x N x D
        enc_out = self.encoder(inp_sub,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len


@TransformerEncoders.register("xfmr_rel")
class RelTransformerEncoder(nn.Module):
    """
    Transformer encoder using relative position encoding
    """

    def __init__(self,
                 input_size,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 radius: int = 128,
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
                                          radius=radius,
                                          dropout=pos_dropout)
        if value_rel_pose:
            self.val_rel_pose = get_xfmr_pose("rel",
                                              embed_dim,
                                              radius=radius,
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

    def _get_rel_encodings(self, inp: th.Tensor) -> EncRetType:
        """
        Return relative embeddings
        """
        nframes = inp.shape[0]
        key_rel_enc = self.key_rel_pose(nframes)
        val_rel_enc = self.val_rel_pose(nframes) if self.val_rel_pose else None
        return key_rel_enc, val_rel_enc

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp_pad: N x Ti x F
            inp_len: N or None
        Return:
            enc_out: N x Ti x D (keep same as aps.asr.base.encoder)
        """
        # inp_sub: N x Ti x D => Ti x N x D
        inp_sub = self.src_proj(inp_pad).transpose(0, 1)
        inp_len = self.src_proj.num_frames(inp_len)
        # src_pad_mask: N x Ti
        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        # rel encodings
        key_rel_enc, value_rel_enc = self._get_rel_encodings(inp_sub)
        # Ti x N x D
        enc_out = self.encoder(inp_sub,
                               src_mask=None,
                               key_rel_pose=key_rel_enc,
                               value_rel_pose=value_rel_enc,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len


class XLTransformerEncoderBase(nn.Module):
    """
    Base class for encoder that use relative position encoding in Xfmr-XL
    """

    def __init__(self,
                 input_size: int,
                 encoder: nn.Module,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Dict = {},
                 att_dim: int = 512,
                 pos_dropout: float = 0.1) -> None:
        super(XLTransformerEncoderBase, self).__init__()
        self.encoder = encoder
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)
        self.sin_pose = get_xfmr_pose("sin", att_dim, dropout=pos_dropout)

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp_pad: N x Ti x F
            inp_len: N or None
        Return:
            enc_out: N x Ti x D (keep same as aps.asr.base.encoder)
        """
        # x_emb: N x Ti x D => Ti x N x D
        inp_sub = self.src_proj(inp_pad).transpose(0, 1)
        inp_len = self.src_proj.num_frames(inp_len)
        # src_pad_mask: N x Ti
        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        # Ti x D
        num_frames, _, _ = inp_sub.shape
        # 2T-1 x D
        sin_pos_enc = self.sin_pose(
            th.arange(-num_frames + 1, num_frames, 1.0, device=inp_sub.device))
        # Ti x N x D
        enc_out = self.encoder(inp_sub,
                               sin_pose=sin_pos_enc,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len


@TransformerEncoders.register("xfmr_xl")
class RelXLTransformerEncoder(XLTransformerEncoderBase):
    """
    Original Transformer-XL encoder
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
        encoder = get_xfmr_encoder("xfmr_xl",
                                   num_layers,
                                   att_dim,
                                   nhead,
                                   dim_feedforward=feedforward_dim,
                                   dropout=att_dropout,
                                   pre_norm=not post_norm,
                                   untie_rel=untie_rel)
        super(RelXLTransformerEncoder,
              self).__init__(input_size,
                             encoder,
                             proj_layer=proj_layer,
                             proj_other_opts=proj_other_opts,
                             att_dim=att_dim,
                             pos_dropout=pos_dropout)


@TransformerEncoders.register("conformer")
class ConformerEncoder(XLTransformerEncoderBase):
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
        encoder = get_xfmr_encoder("conformer",
                                   num_layers,
                                   att_dim,
                                   nhead,
                                   dim_feedforward=feedforward_dim,
                                   dropout=att_dropout,
                                   kernel_size=kernel_size,
                                   pre_norm=False,
                                   untie_rel=untie_rel)
        super(ConformerEncoder, self).__init__(input_size,
                                               encoder,
                                               proj_layer=proj_layer,
                                               proj_other_opts=proj_other_opts,
                                               att_dim=att_dim,
                                               pos_dropout=pos_dropout)
