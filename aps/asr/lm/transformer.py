# wujian@2020

import torch as th
import torch.nn as nn

try:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
except:
    raise ImportError("import Transformer module failed")

from aps.asr.transformer.embedding import IOEmbedding
from aps.asr.transformer.decoder import prep_sub_mask
from aps.asr.base.attention import padding_mask


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
        if embed_size != att_dim:
            raise ValueError("Need embed_size == att_dim")
        self.tgt_embed = IOEmbedding("sparse",
                                     vocab_size,
                                     embed_dim=att_dim,
                                     dropout=pos_dropout)
        encoder_layer = TransformerEncoderLayer(att_dim,
                                                nhead,
                                                dim_feedforward=feedforward_dim,
                                                dropout=att_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # output distribution
        self.dist = nn.Linear(att_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, token, h=None, token_len=None):
        """
        args:
            token: input token sequence, N x T
            h: previous sequence embeddings, T x N x E
            token_len: length of x, N or None
        return:
            output: N x T x V
            h: current sequence embeddings, T x N x E
        """
        # N x T => T x N x V
        t = 0 if h is None else h.shape[0]
        x = self.tgt_embed(token, t=t)
        # h == None: training or eval in time = 0
        h = x if h is None else th.cat([h, x], dim=0)
        # src_pad_mask: N x T
        src_pad_mask = None if token_len is None else (padding_mask(token_len)
                                                       == 1)
        tgt_mask = prep_sub_mask(t + 1, device=x.device)
        # Ti x N x D
        enc_out = self.encoder(h,
                               mask=tgt_mask,
                               src_key_padding_mask=src_pad_mask)
        # N x Ti x D
        enc_out = enc_out.transpose(0, 1)
        # N x Ti x V
        output = self.dist(enc_out)
        return output, h
