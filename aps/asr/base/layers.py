#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Normalize1d(nn.Module):
    """
    Wrapper for BatchNorm1d & LayerNorm
    """

    def __init__(self, name: str, inp_features: int):
        super(Normalize1d, self).__init__()
        name = name.upper()
        if name not in ["BN", "LN"]:
            raise ValueError(f"Unknown type of Normalize1d: {name}")
        if name == "BN":
            self.norm = nn.BatchNorm1d(inp_features, momentum=0)
        else:
            self.norm = nn.LayerNorm(inp_features)
        self.name = name

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x F x T
        Return:
            out (Tensor): N x F x T
        """
        if self.name == "BN":
            # N x F x T => N x T x F
            out = self.norm(inp)
            out = out.transpose(1, 2)
        else:
            inp = inp.transpose(1, 2)
            out = self.norm(inp)
        return out


class TDNN(nn.Module):
    """
    Implement a time delay neural network (TDNN) layer using conv1d operations
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 norm: str = "BN",
                 dropout: float = 0,
                 subsampling: bool = True):
        super(TDNN, self).__init__()
        self.conv1d = nn.Conv1d(inp_features,
                                out_features,
                                kernel_size,
                                stride=stride,
                                dilation=dilation,
                                padding=(dilation * (kernel_size - 1)) // 2)
        self.norm = Normalize1d(norm, out_features)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        self.subsampling_factor = stride if subsampling else 1

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): (N) x T x F
        Return:
            out (Tensor): N x T x O, output of the layer
        """
        if inp.dim() not in [2, 3]:
            raise RuntimeError(
                f"TDNNLayer accepts 2/3D tensor, got {inp.dim()} instead")
        if inp.dim() == 2:
            inp = inp[None, ...]
        _, T, _ = inp.shape
        # N x T x F => N x F x T
        inp = inp.transpose(1, 2)
        # conv
        out = self.conv1d(inp)
        out = out[..., :T // self.subsampling_factor]
        # BN or LN
        out = self.norm(out)
        # ReLU
        out = tf.relu(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class FSMN(nn.Module):
    """
    Implement layer of feedforward sequential memory networks (FSMN)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 proj_features: int,
                 lctx: int = 3,
                 rctx: int = 3,
                 norm: int = "BN",
                 dilation: int = 0,
                 dropout: float = 0):
        super(FSMN, self).__init__()
        self.inp_proj = nn.Linear(inp_features, proj_features, bias=False)
        self.ctx_size = lctx + rctx + 1
        self.ctx_conv = nn.Conv1d(proj_features,
                                  proj_features,
                                  kernel_size=self.ctx_size,
                                  dilation=dilation,
                                  groups=proj_features,
                                  padding=(self.ctx_size - 1) // 2,
                                  bias=False)
        self.out_proj = nn.Linear(proj_features, out_features)
        if norm:
            self.norm = Normalize1d(norm, out_features)
        else:
            self.norm = None
        self.out_drop = nn.Dropout(p=dropout) if dropout > 0 else None

    def forward(
            self,
            inp: th.Tensor,
            memory: Optional[th.Tensor] = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            inp (Tensor): N x T x F, current input
            memory (Tensor or None): N x T x F, memory blocks from previous layer
        Return:
            out (Tensor): N x T x O, output of the layer
            proj (Tensor): N x T x P, new memory block
        """
        if inp.dim() not in [2, 3]:
            raise RuntimeError(f"FSMNLayer expect 2/3D input, got {inp.dim()}")
        if inp.dim() == 2:
            inp = inp[None, ...]
        # N x T x P
        proj = self.inp_proj(inp)
        # N x T x P => N x P x T => N x T x P
        proj = proj.transpose(1, 2)
        # add context
        proj = proj + self.ctx_conv(proj)
        proj = proj.transpose(1, 2)
        # add memory block
        if memory is not None:
            proj = proj + memory
        # N x T x O
        out = self.out_proj(proj)
        # N x O x T
        out = out.transpose(1, 2)
        if self.norm:
            out = self.norm(out)
        out = tf.relu(out)
        if self.out_drop:
            out = self.out_drop(out)
        # N x T x O
        return out, proj


class VariantRNN(nn.Module):
    """
    A custom rnn layer for PyramidEncoder
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 512,
                 project_size: Optional[int] = None,
                 rnn: str = "lstm",
                 layernorm: bool = False,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 add_forward_backward: bool = False):
        super(VariantRNN, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        self.rnn = supported_rnn[RNN](input_size,
                                      hidden_size,
                                      1,
                                      batch_first=True,
                                      bidirectional=bidirectional)
        self.add_forward_backward = add_forward_backward and bidirectional
        self.dropout = nn.Dropout(dropout) if dropout != 0 else None
        if not add_forward_backward and bidirectional:
            hidden_size *= 2
        self.norm = nn.LayerNorm(hidden_size) if layernorm else None
        self.proj = nn.Linear(hidden_size,
                              project_size) if project_size else None

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(self, inp_pad: th.Tensor,
                inp_len: Optional[th.Tensor]) -> th.Tensor:
        """
        Args:
            inp_pad (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x Ti x O
        """
        if inp_len is not None:
            inp_pad = pack_padded_sequence(inp_pad,
                                           inp_len,
                                           batch_first=True,
                                           enforce_sorted=False)
        # extend dim when inference
        else:
            if inp_pad.dim() not in [2, 3]:
                raise RuntimeError("VariantRNN expect input dim as 2 or 3, " +
                                   f"got {inp_pad.dim()}")
            if inp_pad.dim() != 3:
                inp_pad = th.unsqueeze(inp_pad, 0)
        y, _ = self.rnn(inp_pad)
        # N x T x D
        if inp_len is not None:
            y, _ = pad_packed_sequence(y, batch_first=True)
        # add forward & backward
        if self.add_forward_backward:
            f, b = th.chunk(y, 2, dim=-1)
            y = f + b
        # add ln
        if self.norm:
            y = self.norm(y)
        # dropout
        if self.dropout:
            y = self.dropout(y)
        # proj
        if self.proj:
            y = self.proj(y)
        return y
