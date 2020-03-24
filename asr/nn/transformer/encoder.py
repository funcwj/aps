#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from .embedding import IOEmbedding
from ..las.attention import padding_mask


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
                 num_layers=6):
        super(TorchTransformerEncoder, self).__init__()
        self.src_embed = IOEmbedding(input_embed,
                                     input_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout,
                                     other_opts=embed_other_opts)
        encoder_layer = TransformerEncoderLayer(
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