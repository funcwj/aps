#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from aps.asr.transformer.embedding import IOEmbedding
from aps.asr.base.attention import padding_mask
from aps.asr.transformer.xl import _get_transformer_encoder
from aps.asr.transformer.xl import TransformerRelEncoderLayer, TransformerXLEncoderLayer


def support_xfmr_encoder(encoder_name):
    """
    Return transformer decoder
    """
    supported_encoder = {
        "transformer": TorchTransformerEncoder,
        "transformer_rel": RelTransformerEncoder,
        "transformer_rel_xl": RelXLTransformerEncoder
    }
    if encoder_name in supported_encoder:
        return supported_encoder[encoder_name]
    else:
        return None


class ApsTransformerEncoderLayer(TransformerEncoderLayer):
    """
    Transformer encoder with pre-norm
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 pre_norm=False,
                 dropout=0.1,
                 activation="relu"):
        super(ApsTransformerEncoderLayer,
              self).__init__(d_model,
                             nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation)
        self.pre_norm = pre_norm

    def ffn(self, src):
        """
        Get output of the feedforward network
        """
        return self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(src)))))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Support for both pre-norm & post-norm
        """
        inp = src
        if self.pre_norm:
            src = self.norm1(src)
        att = self.self_attn(src,
                             src,
                             src,
                             attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = inp + self.dropout1(att)
        if self.pre_norm:
            src = src + self.dropout2(self.ffn(self.norm2(src)))
        else:
            src = self.norm1(src)
            src = self.norm2(src + self.ffn(src))
        return src


class TorchTransformerEncoder(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 input_size,
                 input_embed="conv2d",
                 embed_other_opts=-1,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 scale_embed=False,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 post_norm=True,
                 num_layers=6):
        super(TorchTransformerEncoder, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     scale_embed=scale_embed,
                                     rel_enc=False,
                                     other_opts=embed_other_opts)
        encoder_layer = ApsTransformerEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.encoder = TransformerEncoder(encoder_layer,
                                          num_layers,
                                          norm=final_norm)
        self.input_embed = input_embed

    def forward(self, x_pad, x_len):
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
                               mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


class RelTransformerEncoder(nn.Module):
    """
    Using relative position encoding
    """

    def __init__(self,
                 input_size,
                 input_embed="conv2d",
                 embed_other_opts=-1,
                 att_dim=512,
                 k_dim=128,
                 nhead=8,
                 feedforward_dim=2048,
                 scale_embed=False,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 post_norm=True,
                 add_value_rel=False,
                 num_layers=6):
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
        self.key_pos = IOEmbedding("sparse", embed_size, embed_dim=embed_dim)
        if add_value_rel:
            self.val_pos = IOEmbedding("sparse",
                                       embed_size,
                                       embed_dim=embed_dim)
        else:
            self.val_pos = None

        encoder_layer = TransformerRelEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout,
            pre_norm=not post_norm)
        final_norm = None if post_norm else nn.LayerNorm(att_dim)
        self.encoder = _get_transformer_encoder(encoder_layer,
                                                num_layers,
                                                norm=final_norm)
        self.input_embed = input_embed
        self.k_dim = k_dim

    def _get_relative_embed(self, inp):
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

    def forward(self, x_pad, x_len):
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
                               key_pos,
                               mask=None,
                               value_pos=value_pos,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len


class RelXLTransformerEncoder(nn.Module):
    """
    Using relative position encoding in Transformer-XL
    """

    def __init__(self,
                 input_size,
                 input_embed="conv2d",
                 embed_other_opts=-1,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 scale_embed=False,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 post_norm=True,
                 untie_rel=True,
                 num_layers=6):
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
        self.encoder = _get_transformer_encoder(encoder_layer,
                                                num_layers,
                                                norm=final_norm)
        self.input_embed = input_embed

    def forward(self, x_pad, x_len):
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
                               p_enc,
                               mask=None,
                               src_key_padding_mask=src_pad_mask)
        return enc_out, x_len
