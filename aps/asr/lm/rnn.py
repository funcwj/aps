# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.init as init
from typing import NoReturn, Union, Tuple, Optional
from aps.asr.base.layer import OneHotEmbedding, PyTorchRNN
from aps.libs import ApsRegisters


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


@ApsRegisters.asr.register("asr@rnn_lm")
class TorchRNNLM(nn.Module):
    """
    A simple Torch RNN LM
    """
    HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]

    def __init__(self,
                 embed_size: int = 256,
                 vocab_size: int = 40,
                 rnn: str = "lstm",
                 dropout: float = 0.2,
                 num_layers: int = 3,
                 hidden_size: int = 512,
                 tie_weights: bool = False) -> None:
        super(TorchRNNLM, self).__init__()
        if embed_size != vocab_size:
            self.embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.embed = OneHotEmbedding(vocab_size)
        # uni-directional RNNs
        self.pred = PyTorchRNN(rnn,
                               embed_size,
                               hidden_size,
                               num_layers=num_layers,
                               dropout=dropout,
                               bidirectional=False)
        # output distribution
        self.dist = nn.Linear(hidden_size, vocab_size)
        self.embed_drop = nn.Dropout(p=dropout)
        self.pred_drop = nn.Dropout(p=dropout)
        self.vocab_size = vocab_size

        self.init_weights()

        # tie_weights
        if tie_weights and embed_size == hidden_size:
            self.dist.weight = self.embed.weight

    def init_weights(self, initrange: float = 0.1) -> NoReturn:
        init.zeros_(self.dist.bias)
        init.uniform_(self.dist.weight, -initrange, initrange)
        init.uniform_(self.embed.weight, -initrange, initrange)

    def forward(self,
                token: th.Tensor,
                h: Optional[HiddenType] = None,
                token_len=Optional[th.Tensor]) -> Tuple[th.Tensor, HiddenType]:
        """
        Args:
            token: input token sequence, N x T
            h: hidden state from previous time step
            token_len: length of x, N or None
        Return:
            output: N x T x V
            h: hidden state from current time step
        """
        if h is not None:
            h = repackage_hidden(h)
        # N x T => N x T x V
        inp_emb = self.embed_drop(self.embed(token))
        # N x T x H
        out, h = self.pred(inp_emb, h)
        # N x T x V
        out = self.dist(self.pred_drop(out))
        return out, h
