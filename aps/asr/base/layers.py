#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union, NoReturn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]


class OneHotEmbedding(nn.Module):
    """
    Onehot embedding layer
    """

    def __init__(self, vocab_size: int):
        super(OneHotEmbedding, self).__init__()
        self.vocab_size = vocab_size

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}"

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): ...
        Return
            e (Tensor): ... x V
        """
        S = list(x.shape) + [self.vocab_size]
        # ... x V
        H = th.zeros(S, dtype=th.float32, device=x.device)
        # set one
        H = H.scatter(-1, x[..., None], 1)
        return H


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
            self.norm = nn.BatchNorm1d(inp_features)
        else:
            self.norm = nn.LayerNorm(inp_features)
        self.name = name

    def __repr__(self) -> str:
        return str(self.norm)

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


def PyTorchRNN(mode: str,
               input_size: int,
               hidden_size: int,
               num_layers: int = 1,
               bias: bool = True,
               dropout: float = 0.,
               bidirectional: bool = False) -> nn.Module:
    """
    Wrapper for PyTorch RNNs (LSTM, GRU, RNN_TANH, RNN_RELU)
    """
    supported_rnn = {
        "RNN_TANH": nn.RNN,
        "RNN_RELU": nn.ReLU,
        "GRU": nn.GRU,
        "LSTM": nn.LSTM
    }
    mode = mode.upper()
    if mode not in supported_rnn:
        raise ValueError(f"Unsupported RNNs: {mode}")
    kwargs = {
        "bias": bias,
        "dropout": dropout,
        "batch_first": True,
        "bidirectional": bidirectional
    }
    if mode in ["GRU", "LSTM"]:
        return supported_rnn[mode](input_size, hidden_size, num_layers,
                                   **kwargs)
    elif mode == "RNN_TANH":
        return supported_rnn[mode](input_size,
                                   hidden_size,
                                   num_layers,
                                   nonlinearity="tanh",
                                   **kwargs)
    else:
        return supported_rnn[mode](input_size,
                                   hidden_size,
                                   num_layers,
                                   nonlinearity="relu",
                                   **kwargs)


class InputDropoutRNN(nn.Module):
    """
    Avoid applying dropout along time axis (for decoders)
    """

    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 dropout: float = 0,
                 bidirectional: bool = False):
        super(InputDropoutRNN, self).__init__()
        self.inp_dropout = nn.Dropout(p=dropout)
        self.rnn_pytorch = PyTorchRNN(mode,
                                      input_size,
                                      hidden_size,
                                      num_layers=1,
                                      dropout=0,
                                      bidirectional=bidirectional,
                                      bias=bias)

    def forward(
            self,
            inp: th.Tensor,
            hx: Optional[HiddenType] = None,
            inp_len: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, HiddenType]:
        """
        Args:
            inp (Tensor): N x T x D
        Return:
            out (Tensor): N x T x H
            hx (Tensor or [Tensor, Tensor]): 1/2 x N x H
        """
        inp = self.inp_dropout(inp)
        if inp_len is not None:
            inp = pack_padded_sequence(inp,
                                       inp_len,
                                       batch_first=True,
                                       enforce_sorted=False)
        out, hx = self.rnn_pytorch(inp, hx)
        if inp_len is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
        return out, hx


class DropoutRNN(nn.Module):
    """
    Stack of InputDropoutRNN
    """

    def __init__(self,
                 mode: str,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float = 0,
                 bidirectional: bool = False) -> None:
        super(DropoutRNN, self).__init__()
        self.layers = nn.ModuleList([
            InputDropoutRNN(mode,
                            input_size if i == 0 else hidden_size,
                            hidden_size,
                            dropout=0 if i == 0 else dropout,
                            bidirectional=bidirectional)
            for i in range(num_layers)
        ])
        self.factor = 2 if bidirectional else 1

    def forward(
            self,
            inp: th.Tensor,
            hx: Optional[HiddenType] = None) -> Tuple[th.Tensor, HiddenType]:
        """
        Args:
            inp (Tensor): N x T x D
        Return:
            out (Tensor): N x T x H
        """
        states = []
        for index, layer in enumerate(self.layers):
            # state: 1 x N x H
            index = index * self.factor
            if hx is None:
                state = None
            elif isinstance(hx, th.Tensor):
                state = hx[index:index + self.factor]
            else:
                h, c = hx
                state = (h[index:index + self.factor],
                         c[index:index + self.factor])
            inp, state = layer(inp, hx=state)
            states.append(state)
        # for RNN/GRU
        if isinstance(states[0], th.Tensor):
            states = th.cat(states, 0)
        else:
            h = th.cat([s[0] for s in states], 0)
            c = th.cat([s[1] for s in states], 0)
            states = (h, c)
        return inp, states


class Conv1d(nn.Module):
    """
    Time delay neural network (TDNN) layer using conv1d operations
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 norm: str = "BN",
                 dropout: float = 0):
        super(Conv1d, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(inp_features,
                              out_features,
                              kernel_size,
                              stride=stride,
                              dilation=dilation,
                              padding=padding)
        self.norm = Normalize1d(norm, out_features)
        self.drop = nn.Dropout(p=dropout)
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding

    def compute_outp_dim(self, dim: th.Tensor) -> th.Tensor:
        """
        Compute output dimention
        """
        return (dim + 2 * self.padding - self.dilation *
                (self.kernel_size - 1) - 1) // self.stride + 1

    def check_args(self, inp: th.Tensor) -> NoReturn:
        """
        Check args
        """
        if inp.dim() != 3:
            raise RuntimeError(
                f"TDNN expects 3D tensor, got {inp.dim()} instead")

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): (N) x T x F
        Return:
            out (Tensor): N x T x O, output of the layer
        """
        self.check_args(inp)
        # N x T x F => N x F x T
        inp = inp.transpose(1, 2)
        # conv & norm
        out = self.norm(self.conv(inp))
        # ReLU & dropout
        out = self.drop(tf.relu(out))
        return out


class Conv2d(nn.Module):
    """
    A wrapper for conv2d layer, with batchnorm & ReLU
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 2,
                 padding: Union[int, Tuple[int]] = 0):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=True)
        self.norm = nn.BatchNorm2d(out_channels)

        def int2tuple(inp):
            return (inp, inp) if isinstance(inp, int) else inp

        self.kernel_size = int2tuple(kernel_size)
        self.stride = int2tuple(stride)
        self.padding = int2tuple(padding)

    def compute_outp_dim(self, dim: th.Tensor, axis: int) -> th.Tensor:
        """
        Compute output dimention
        """
        return (dim + 2 * self.padding[axis] -
                self.kernel_size[axis]) // self.stride[axis] + 1

    def check_args(self, inp: th.Tensor) -> NoReturn:
        """
        Check args
        """
        if inp.dim() not in [3, 4]:
            raise RuntimeError(
                f"Conv2d expects 3/4D tensor, got {inp.dim()} instead")

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T x F
        Return:
            out (Tensor): N x C' x T' x F'
        """
        self.check_args(inp)
        out = self.norm(self.conv(inp[:, None] if inp.dim() == 3 else inp))
        return tf.relu(out)


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
        self.out_drop = nn.Dropout(p=dropout)

    def check_args(self, inp: th.Tensor) -> NoReturn:
        """
        Check args
        """
        if inp.dim() not in [2, 3]:
            raise RuntimeError(f"FSMN expects 2/3D input, got {inp.dim()}")

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
        self.check_args(inp)
        # N x T x P
        proj = self.inp_proj(inp[None, ...] if inp.dim() == 2 else inp)
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
        out = self.out_drop(tf.relu(out))
        # N x T x O
        return out, proj


class VariantRNN(nn.Module):
    """
    A custom rnn layer to support other features
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int = 512,
                 rnn: str = "lstm",
                 layernorm: bool = False,
                 project: Optional[int] = None,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 add_forward_backward: bool = False):
        super(VariantRNN, self).__init__()
        self.rnn = InputDropoutRNN(rnn,
                                   input_size,
                                   hidden_size,
                                   dropout=dropout,
                                   bidirectional=bidirectional)
        self.add_forward_backward = add_forward_backward and bidirectional
        if bidirectional and not add_forward_backward:
            hidden_size *= 2
        self.norm = nn.LayerNorm(hidden_size) if layernorm else None
        self.proj = nn.Linear(hidden_size, project) if project else None

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x Ti x O
        """
        out, _ = self.rnn(inp, hx=None, inp_len=inp_len)
        # add forward & backward
        if self.add_forward_backward:
            fp, bp = th.chunk(out, -1, dim=-1)
            out = fp + bp
        # add ln
        if self.norm:
            out = self.norm(out)
        # proj
        if self.proj:
            out = self.proj(out)
        return out
