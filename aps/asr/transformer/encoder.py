#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Tuple
from aps.asr.transformer.embedding import IOEmbedding
from aps.asr.base.attention import padding_mask
from aps.asr.transformer.impl import TransformerTorchEncoderLayer, TransformerRelEncoderLayer, TransformerXLEncoderLayer
from aps.asr.transformer.impl import ApsTransformerEncoder
from aps.libs import Register

TransformerEncoders = Register("xfmr_encoder")


def support_xfmr_encoder(encoder_name):
    """
    Return transformer decoder
    """
    if encoder_name in TransformerEncoders:
        return TransformerEncoders[encoder_name]
    else:
        return None


@TransformerEncoders.register("transformer")
class TorchTransformerEncoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 input_size: int,
                 input_embed: str = "conv2d",
                 embed_other_opts: int = -1,
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
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     scale_embed=scale_embed,
                                     rel_enc=False,
                                     pos_enc=pos_enc,
                                     other_opts=embed_other_opts)
        encoder_layer = TransformerTorchEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.encoder = ApsTransformerEncoder(encoder_layer,
                                             num_layers,
                                             norm=final_norm)
        self.input_embed = input_embed

    def forward(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        if self.input_embed[:4] == "conv" and x_len is not None:
            x_len = x_len // 4
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.src_embed(x_pad)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


@TransformerEncoders.register("transformer_rel")
class RelTransformerEncoder(nn.Module):
    """
    Using relative position encoding
    """

    def __init__(self,
                 input_size,
                 input_embed="conv2d",
                 embed_other_opts: int = -1,
                 att_dim: int = 512,
                 k_dim: int = 128,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 post_norm: bool = True,
                 add_value_rel: bool = False,
                 num_layers: int = 6) -> None:
        super(RelTransformerEncoder, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     scale_embed=scale_embed,
                                     pos_enc=False,
                                     other_opts=embed_other_opts)
        embed_size = 2 * k_dim + 1
        embed_dim = att_dim // nhead
        self.key_pos = IOEmbedding("sparse",
                                   embed_size,
                                   embed_dim=embed_dim,
                                   pos_enc=False)
        if add_value_rel:
            self.val_pos = IOEmbedding("sparse",
                                       embed_size,
                                       embed_dim=embed_dim,
                                       pos_enc=False)
        else:
            self.val_pos = None

        encoder_layer = TransformerRelEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.encoder = ApsTransformerEncoder(encoder_layer,
                                             num_layers,
                                             norm=final_norm)
        self.input_embed = input_embed
        self.k_dim = k_dim

    def _get_relative_embed(
            self, inp: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Return relative embeddings
        """
        seq_vec = th.arange(inp.shape[0], device=inp.device)
        # T x T
        seq_mat = th.clamp(seq_vec[:, None] - seq_vec[None, :],
                           max=self.k_dim,
                           min=-self.k_dim) + self.k_dim
        # to int64
        seq_mat = seq_mat.to(th.int64)
        key_pos = self.key_pos(seq_mat)
        if self.val_pos:
            val_pos = self.val_pos(seq_mat)
        else:
            val_pos = None
        return key_pos, val_pos

    def forward(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        if self.input_embed[:4] == "conv" and x_len is not None:
            x_len = x_len // 4
        # x_emb: N x Ti x D => Ti x N x D
        x_emb = self.src_embed(x_pad)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # rel embeddings
        key_pos, value_pos = self._get_relative_embed(x_emb)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               src_mask=None,
                               key_pos=key_pos,
                               value_pos=value_pos,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


@TransformerEncoders.register("transformer_rel_xl")
class RelXLTransformerEncoder(nn.Module):
    """
    Using relative position encoding in Transformer-XL
    """

    def __init__(self,
                 input_size: int,
                 input_embed: str = "conv2d",
                 embed_other_opts: int = -1,
                 att_dim: int = 512,
                 nhead: int = 8,
                 feedforward_dim: int = 2048,
                 scale_embed: bool = False,
                 pos_dropout: float = 0.1,
                 att_dropout: float = 0.1,
                 post_norm: bool = True,
                 untie_rel: bool = True,
                 num_layers: int = 6) -> None:
        super(RelXLTransformerEncoder, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     scale_embed=scale_embed,
                                     rel_enc=True,
                                     other_opts=embed_other_opts)
        if not untie_rel:
            rel_u = nn.Parameter(th.Tensor(nhead, att_dim // nhead))
            rel_v = nn.Parameter(th.Tensor(nhead, att_dim // nhead))
            nn.init.normal_(rel_u, std=0.02)
            nn.init.normal_(rel_v, std=0.02)
        else:
            rel_u, rel_v = None, None
        encoder_layer = TransformerXLEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm,
            rel_u=rel_u,
            rel_v=rel_v)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.encoder = ApsTransformerEncoder(encoder_layer,
                                             num_layers,
                                             norm=final_norm)
        self.input_embed = input_embed

    def forward(
            self, x_pad: th.Tensor, x_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            x_pad: N x Ti x F
            x_len: N or None
        Return:
            enc_out: Ti x N x D
        """
        if self.input_embed[:4] == "conv" and x_len is not None:
            x_len = x_len // 4
        # x_emb: N x Ti x D => Ti x N x D
        # p_enc: Ti x D, sin encodings
        p_enc, x_emb = self.src_embed(x_pad)
        # src_pad_mask: N x Ti
        src_pad_mask = None if x_len is None else (padding_mask(x_len) == 1)
        # Ti x N x D
        enc_out = self.encoder(x_emb,
                               sin_pos_enc=p_enc,
                               src_mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len
