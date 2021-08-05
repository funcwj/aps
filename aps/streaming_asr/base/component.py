#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from aps.asr.base.component import PyTorchRNN, Normalize1d, rnn_output_nonlinear
from typing import Optional, Tuple


class PyTorchNormLSTM(nn.Module):
    """
    LSTM with layer normalization
    """
    LSTMHiddenType = Optional[Tuple[th.Tensor, th.Tensor]]

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 512,
                 proj_size: Optional[int] = -1,
                 non_linear: str = "relu",
                 dropout: float = 0.0):
        super(PyTorchNormLSTM, self).__init__()
        self.lstm = PyTorchRNN("lstm",
                               input_size,
                               hidden_size,
                               num_layers=1,
                               dropout=0,
                               proj_size=proj_size,
                               bidirectional=False)
        self.norm = Normalize1d("ln",
                                proj_size if proj_size > 0 else hidden_size)
        self.non_linear = rnn_output_nonlinear[non_linear]
        self.drop = nn.Dropout(p=dropout)

    def forward(self,
                inp: th.Tensor,
                hx: LSTMHiddenType = None) -> Tuple[th.Tensor, LSTMHiddenType]:
        out, hx = self.lstm(inp, hx)
        out = self.norm(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return self.drop(out), hx
