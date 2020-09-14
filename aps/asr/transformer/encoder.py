#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from aps.asr.transformer.embedding import IOEmbedding
from aps.asr.base.attention import padding_mask


class PreNormTransformerEncoderLayer(TransformerEncoderLayer):
    """
    Transformer encoder with pre-norm
    """

    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(PreNormTransformerEncoderLayer,
              self).__init__(d_model,
                             nhead,
                             dim_feedforward=dim_feedforward,
                             dropout=dropout,
                             activation=activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Code for Post-Norm Transformer are:
        src2 = self.self_attn(src,
                              src,
                              src,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
        """
        src1 = self.norm1(src)
        src2 = self.self_attn(src1,
                              src1,
                              src1,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src3 = self.norm2(src)
        # PositionwiseFF
        src4 = self.linear2(self.dropout(self.activation(self.linear1(src3))))
        src = src + self.dropout2(src4)
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
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 post_norm=True,
                 num_layers=6):
        super(TorchTransformerEncoder, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     other_opts=embed_other_opts)
        if post_norm:
            encoder_layer = TransformerEncoderLayer(
                att_dim,
                nhead,
                dim_feedforward=feedforward_dim,
                dropout=att_dropout)
        else:
            encoder_layer = PreNormTransformerEncoderLayer(
                att_dim,
                nhead,
                dim_feedforward=feedforward_dim,
                dropout=att_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.input_embed = input_embed

    def forward(self, x_pad, x_len):
        """
        args:
            x_pad: N x Ti x F
            x_len: N or None
        return:
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
