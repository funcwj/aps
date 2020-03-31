# wujian@2019

import torch as th
import torch.nn as nn

from ..las.decoder import OneHotEmbedding


def repackage_hidden(h):
    """
    detach variable from graph
    """
    if isinstance(h, th.Tensor):
        return h.detach()
    elif isinstance(h, tuple):
        return tuple(repackage_hidden(v) for v in h)
    elif isinstance(h, list):
        return list(repackage_hidden(v) for v in h)
    else:
        raise TypeError(f"Unsupport type: {type(h)}")


class TorchLM(nn.Module):
    """
    A simple Torch RNN LM
    """
    def __init__(self,
                 embed_size=256,
                 vocab_size=40,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.2):
        super(TorchLM, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if embed_size != vocab_size:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        # uni-directional RNNs
        self.pred = supported_rnn[RNN](embed_size,
                                       rnn_hidden,
                                       rnn_layers,
                                       batch_first=True,
                                       dropout=rnn_dropout,
                                       bidirectional=False)
        # output distribution
        self.dist = nn.Linear(rnn_hidden, vocab_size)

    def forward(self, x, h=None, detach_h=True):
        """
        args:
            x: N x T
            h: hidden state from previous time step
        return:
            y: N x T x V
        """
        if h is not None and detach_h:
            h = repackage_hidden(h)
        # N x T => N x T x V
        x = self.vocab_embed(x)
        y, h = self.pred(x, h)
        # N x T x V
        y = self.dist(y)
        return y, h
