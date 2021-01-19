#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Dict, Tuple
from aps.asr.base.attention import padding_mask
from aps.asr.base.encoder import EncRetType
from aps.asr.xfmr.impl import get_xfmr_encoder
from aps.asr.xfmr.pose import get_xfmr_pose
from aps.asr.xfmr.utils import get_xfmr_proj
from aps.libs import Register

TransformerEncoders = Register("xfmr_encoder")
ProjRetType = Tuple[th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]


def support_xfmr_encoder(encoder_name: str) -> Optional[nn.Module]:
    """
    Return transformer encoder
    """
    if encoder_name in TransformerEncoders:
        return TransformerEncoders[encoder_name]
    else:
        return None


class TransformerEncoderBase(nn.Module):
    """
    Base class for Transformer based encoders
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512):
        super(TransformerEncoderBase, self).__init__()
        self.src_proj = get_xfmr_proj(proj_layer, input_size, att_dim,
                                      proj_other_opts)

    def proj(self, inp_pad: th.Tensor,
             inp_len: Optional[th.Tensor]) -> ProjRetType:
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
        inp_len = self.src_proj.num_frames(inp_len)
        enc_inp = self.src_proj(inp_pad)
        src_pad_mask = None if inp_len is None else (padding_mask(inp_len) == 1)
        return enc_inp, inp_len, src_pad_mask


@TransformerEncoders.register("xfmr")
class TorchTransformerEncoder(TransformerEncoderBase):
    """
    The standard Transformer encoder
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 post_norm: bool = True,
                 num_layers: int = 6) -> None:
        super(TorchTransformerEncoder,
              self).__init__(input_size,
                             proj_layer=proj_layer,
                             proj_other_opts=proj_other_opts,
                             att_dim=att_dim)
        self.abs_pos_enc = get_xfmr_pose("inp_sin",
                                         att_dim,
                                         dropout=pos_dropout,
                                         scale_embed=scale_embed)
        self.encoder = get_xfmr_encoder("xfmr",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        att_dropout=att_dropout,
                                        ffn_dropout=ffn_dropout,
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
        enc_inp, inp_len, src_pad_mask = self.proj(inp_pad, inp_len)
        # enc_inp: N x Ti x D => Ti x N x D
        enc_inp = self.abs_pos_enc(enc_inp)
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len


@TransformerEncoders.register("xfmr_rel")
class RelTransformerEncoder(TransformerEncoderBase):
    """
    Transformer encoder using relative position encoding
    """

    def __init__(self,
                 input_size,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512,
                 radius: int = 128,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 post_norm: bool = True,
                 num_layers: int = 6) -> None:
        super(RelTransformerEncoder,
              self).__init__(input_size,
                             proj_layer=proj_layer,
                             proj_other_opts=proj_other_opts,
                             att_dim=att_dim)
        self.key_rel_pose = get_xfmr_pose("rel",
                                          att_dim // nhead,
                                          radius=radius,
                                          dropout=pos_dropout)
        self.encoder = get_xfmr_encoder("xfmr_rel",
                                        num_layers,
                                        att_dim,
                                        nhead,
                                        dim_feedforward=feedforward_dim,
                                        att_dropout=att_dropout,
                                        ffn_dropout=ffn_dropout,
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
        enc_inp, inp_len, src_pad_mask = self.proj(inp_pad, inp_len)
        # enc_inp: N x Ti x D => Ti x N x D
        enc_inp = enc_inp.transpose(0, 1)
        # rel encodings: 2Ti-1 x D
        key_rel_enc = self.key_rel_pose(
            th.arange(-enc_inp.shape[0] + 1,
                      enc_inp.shape[0],
                      device=enc_inp.device))
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
                               src_mask=None,
                               key_rel_pose=key_rel_enc,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        return enc_out.transpose(0, 1), inp_len


class XLTransformerEncoderBase(TransformerEncoderBase):
    """
    Base class for encoders using relative position encoding in Transformer-XL
    """

    def __init__(self,
                 input_size: int,
                 encoder: nn.Module,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512,
                 pos_dropout: float = 0.1) -> None:
        super(XLTransformerEncoderBase,
              self).__init__(input_size,
                             proj_layer=proj_layer,
                             proj_other_opts=proj_other_opts,
                             att_dim=att_dim)
        self.encoder = encoder
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
        enc_inp, inp_len, src_pad_mask = self.proj(inp_pad, inp_len)
        # enc_inp: N x Ti x D => Ti x N x D
        enc_inp = enc_inp.transpose(0, 1)
        # 2Ti-1 x D
        sin_pos_enc = self.sin_pose(
            th.arange(0, 2 * enc_inp.shape[0] - 1, 1.0, device=enc_inp.device))
        # Ti x N x D
        enc_out = self.encoder(enc_inp,
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
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 post_norm: bool = True,
                 untie_rel: bool = True,
                 num_layers: int = 6) -> None:
        encoder = get_xfmr_encoder("xfmr_xl",
                                   num_layers,
                                   att_dim,
                                   nhead,
                                   dim_feedforward=feedforward_dim,
                                   att_dropout=att_dropout,
                                   ffn_dropout=ffn_dropout,
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
    Conformer encoder proposed by Google
        Conformer: Convolution-augmented Transformer for Speech Recognition
    """

    def __init__(self,
                 input_size: int,
                 proj_layer: str = "conv2d",
                 proj_other_opts: Optional[Dict] = None,
                 att_dim: int = 512,
                 nhead: int = 8,
                 untie_rel: bool = True,
                 feedforward_dim: int = 2048,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 kernel_size: int = 16,
                 num_layers: int = 6) -> None:
        encoder = get_xfmr_encoder("conformer",
                                   num_layers,
                                   att_dim,
                                   nhead,
                                   dim_feedforward=feedforward_dim,
                                   att_dropout=att_dropout,
                                   ffn_dropout=ffn_dropout,
                                   kernel_size=kernel_size,
                                   pre_norm=False,
                                   untie_rel=untie_rel)
        super(ConformerEncoder, self).__init__(input_size,
                                               encoder,
                                               proj_layer=proj_layer,
                                               proj_other_opts=proj_other_opts,
                                               att_dim=att_dim,
                                               pos_dropout=pos_dropout)
