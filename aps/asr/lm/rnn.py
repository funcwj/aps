# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import NoReturn, Union, Tuple, Optional
from aps.asr.base.layers import OneHotEmbedding, PyTorchRNN, DropoutRNN
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


@ApsRegisters.asr.register("rnn_lm")
class TorchRNNLM(nn.Module):
    """
    A simple Torch RNN LM
    """
    HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]

    def __init__(self,
                 embed_size: int = 256,
                 vocab_size: int = 40,
                 rnn: str = "lstm",
                 dropout_on: str = "state",
                 rnn_layers: int = 3,
                 rnn_hidden: int = 512,
                 rnn_dropout: float = 0.2,
                 tie_weights: bool = False) -> None:
        super(TorchRNNLM, self).__init__()
        dropout_rnn_cls = {"state": PyTorchRNN, "input": DropoutRNN}
        if dropout_on not in dropout_rnn_cls:
            raise ValueError(f"Unsupported dropout_on: {dropout_on}")
        self.vocab_drop = nn.Dropout(rnn_dropout)
        if embed_size != vocab_size:
            self.vocab_embed = nn.Embedding(vocab_size, embed_size)
        else:
            self.vocab_embed = OneHotEmbedding(vocab_size)
        # uni-directional RNNs
        self.pred = dropout_rnn_cls[dropout_on](rnn,
                                                embed_size,
                                                rnn_hidden,
                                                rnn_layers,
                                                dropout=rnn_dropout,
                                                bidirectional=False)
        # output distribution
        self.dist = nn.Linear(rnn_hidden, vocab_size)
        self.vocab_size = vocab_size

        self.init_weights()

        # tie_weights
        if tie_weights and embed_size == rnn_hidden:
            self.dist.weight = self.vocab_embed.weight

    def init_weights(self, initrange: float = 0.1) -> NoReturn:
        self.vocab_embed.weight.data.uniform_(-initrange, initrange)
        self.dist.bias.data.zero_()
        self.dist.weight.data.uniform_(-initrange, initrange)

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
        x = self.vocab_embed(token)
        x = self.vocab_drop(x)

        # N x T x H
        y, h = self.pred(x, h)
        # N x T x V
        output = self.dist(y)
        return output, h
