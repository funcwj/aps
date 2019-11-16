# wujian@2019

import torch as th
import torch.nn as nn

from ..las.decoder import OneHotEmbedding


class RNNLM(nn.Module):
    """
    A simple RNN language model
    """
    def __init__(self,
                 embed_size=256,
                 vocab_size=40,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0):
        super(RNNLM, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if embed_size != vocab_size:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        # uni-directional RNNs
        self.rnns = supported_rnn[RNN](embed_size,
                                       hidden_size,
                                       num_layers,
                                       batch_first=True,
                                       dropout=dropout,
                                       bidirectional=False)
        # output distribution
        self.dist = nn.Linear(hidden_size, vocab_size)

    def repackage_hidden(self, h):
        """
        Detach variable from graph
        """
        if isinstance(h, th.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, x, h=None):
        """
        args:
            x: N x T
            h: hidden state from previous time step
        return:
            y: N x T x V
        """
        if h is not None:
            h = self.repackage_hidden(h)
        # N x T => N x T x V
        x = self.vocab_embed(x)
        y, h = self.rnns(x, h)
        # N x T x V
        y = self.dist(y)
        return y, h
