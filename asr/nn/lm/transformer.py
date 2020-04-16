# wujian@2020

import torch as th
import torch.nn as nn

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from ..transformer.embedding import IOEmbedding
from ..las.attention import padding_mask


class TorchTransformerLM(nn.Module):
    """
    Torch Transformer LM
    """
    def __init__(self,
                 embed_size=256,
                 vocab_size=40,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 num_layers=6):
        super(TorchTransformerLM, self).__init__()
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=embed_size,
                                     dropout=pos_dropout)
        encoder_layer = TransformerEncoderLayer(
            att_dim,
            nhead,
            dim_feedforward=feedforward_dim,
            dropout=att_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # output distribution
        self.dist = nn.Linear(att_dim, vocab_size)

    def forward(self, x, h=None, xlen=None):
        """
        args:
            x: N x T
            h: hidden state (None here)
            xlen: N or None
        return:
            y: N x T x V
        """
        # N x T => T x N x V
        x = self.tgt_embed(x)
        # src_pad_mask: N x T
        src_pad_mask = None if xlen is None else (padding_mask(xlen) == 1)
        # Ti x N x D
        enc_out = self.encoder(x, mask=None, src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        enc_out = enc_out.transpose(0, 1)
        # N x Ti x V
        y = self.dist(enc_out)
        return y, h