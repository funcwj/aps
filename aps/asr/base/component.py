#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union
from distutils.version import LooseVersion
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from aps.const import TORCH_VERSION

HiddenType = Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]

rnn_output_nonlinear = {
    "relu": th.relu,
    "sigmoid": th.sigmoid,
    "tanh": th.tanh,
    "none": None,
}


def var_len_rnn_forward(rnn_impl: nn.Module,
                        inp: th.Tensor,
                        inp_len: Optional[th.Tensor] = None,
                        enforce_sorted: bool = False,
                        add_forward_backward: bool = False) -> th.Tensor:
    """
    Forward of the RNN with variant length input
    Args:
        inp (Tensor): N x T x D
        inp_len (Tensor or None): N
    Return:
        out (Tensor): N x T x H
    """
    if inp.dim() != 3:
        raise ValueError(
            f"RNN forward needs 3D tensor, got {inp.dim()} instead")
    if inp_len is not None:
        inp = pack_padded_sequence(
            inp,
            inp_len
            if TORCH_VERSION < LooseVersion("1.7") else inp_len.tolist(),
            batch_first=True,
            enforce_sorted=enforce_sorted)
    out, _ = rnn_impl(inp)
    if inp_len is not None:
        out, _ = pad_packed_sequence(out, batch_first=True)
    if add_forward_backward:
        prev, last = th.chunk(out, 2, dim=-1)
        out = prev + last
    return out


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
            self.norm = nn.GroupNorm(1, inp_features)

    def __repr__(self) -> str:
        return str(self.norm)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x F
        Return:
            out (Tensor): N x T x F
        """
        # N x T x F => N x F x T
        inp = inp.transpose(1, 2)
        out = self.norm(inp)
        out = out.transpose(1, 2)
        return out


class Normalize2d(nn.Module):
    """
    Wrapper for BatchNorm2d & InstanceNorm2d
    """

    def __init__(self, name: str, inp_features: int):
        super(Normalize2d, self).__init__()
        name = name.upper()
        if name not in ["BN", "IN"]:
            raise ValueError(f"Unknown type of Normalize2d: {name}")
        if name == "BN":
            self.norm = nn.BatchNorm2d(inp_features)
        else:
            self.norm = nn.InstanceNorm2d(inp_features)

    def __repr__(self) -> str:
        return str(self.norm)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T x F
        Return:
            out (Tensor): N x C x T x F
        """
        return self.norm(inp)


def PyTorchRNN(mode: str,
               input_size: int,
               hidden_size: int,
               num_layers: int = 1,
               bias: bool = True,
               dropout: float = 0.,
               proj_size: int = -1,
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
        # add proj_size if needed
        if mode == "LSTM" and TORCH_VERSION >= LooseVersion(
                "1.8") and proj_size > 0:
            kwargs["proj_size"] = proj_size
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


class Conv1d(nn.Module):
    """
    Time delay neural network (TDNN) layer using conv1d operations (... -> Conv1d -> Norm -> ReLU -> Dropout -> ...)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 dilation: int = 1,
                 norm: str = "BN",
                 dropout: float = 0,
                 for_streaming: bool = False):
        super(Conv1d, self).__init__()
        # we padding 0 by hand
        if for_streaming:
            padding = 0
        else:
            padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv1d(inp_features,
                              out_features,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
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
        return th.div(dim + 2 * self.padding - self.dilation *
                      (self.kernel_size - 1) - 1,
                      self.stride,
                      rounding_mode="trunc") + 1

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x F
        Return:
            out (Tensor): N x T x O, output of the layer
        """
        # N x T x F => N x F x T
        inp = inp.transpose(1, 2)
        out = self.conv(inp)
        # N x T x F
        out = out.transpose(1, 2)
        # norm & ReLU & dropout
        out = self.drop(tf.relu(self.norm(out)))
        return out


class Conv2d(nn.Module):
    """
    A simple Conv2d block (... -> Conv2d -> Norm -> ReLU -> ...)
    """
    Conv2dParam = Union[int, Tuple[int, int]]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Conv2dParam = 3,
                 stride: Conv2dParam = 2,
                 dilation: Conv2dParam = 1,
                 norm: str = "BN",
                 for_streaming: bool = False):
        super(Conv2d, self).__init__()

        def int2tuple(inp):
            return (inp, inp) if isinstance(inp, int) else inp

        kernel_size = int2tuple(kernel_size)
        dilation = int2tuple(dilation)

        padding = tuple(
            (d * (k - 1)) // 2 for d, k in zip(dilation, kernel_size))
        # disable time axis
        if for_streaming:
            padding = (0, padding[-1])
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation)
        self.norm = Normalize2d(norm, out_channels)
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.stride = int2tuple(stride)

    def compute_outp_dim(self, dim: th.Tensor, axis: int) -> th.Tensor:
        """
        Compute output dimention
        """
        return th.div(dim + 2 * self.padding[axis] -
                      self.dilation[axis] * self.kernel_size[axis],
                      self.stride[axis],
                      rounding_mode="trunc") + 1

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T x F
        Return:
            out (Tensor): N x C' x T' x F'
        """
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
                 norm: str = "BN",
                 dilation: int = 0,
                 dropout: float = 0.0,
                 for_streaming: bool = False):
        super(FSMN, self).__init__()
        self.inp_proj = nn.Linear(inp_features, proj_features, bias=False)
        self.ctx_conv = nn.Conv1d(proj_features,
                                  proj_features,
                                  kernel_size=lctx + rctx + 1,
                                  dilation=dilation,
                                  groups=proj_features,
                                  padding=0,
                                  bias=False)
        self.out_proj = nn.Linear(proj_features, out_features)
        if norm == "none":
            self.out_norm = None
        else:
            self.out_norm = nn.Sequential(Normalize1d(norm, out_features),
                                          nn.ReLU(), nn.Dropout(p=dropout))
        self.lctx, self.rctx = lctx, rctx
        self.streaming = for_streaming

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
        # N x T x P
        proj = self.inp_proj(inp[None, ...] if inp.dim() == 2 else inp)
        # N x T x P => N x P x T => N x T x P
        proj = proj.transpose(1, 2)
        # pad context
        if not self.streaming:
            # NOTE: ctx.shape[1] == proj.shape[1]
            proj_pad = tf.pad(proj, (self.lctx, self.rctx), "constant", 0.0)
            ctx = self.ctx_conv(proj_pad)
        else:
            # NOTE: ctx.shape[1] != proj.shape[1]
            ctx = self.ctx_conv(proj)
            if self.rctx > 0:
                proj = proj[..., self.lctx:-self.rctx]
                if memory is not None:
                    memory = memory[:, self.lctx:-self.rctx]
            else:
                proj = proj[..., self.lctx:]
                if memory is not None:
                    memory = memory[:, self.lctx:]
        # add context
        proj = proj + ctx
        proj = proj.transpose(1, 2)
        # add memory block
        if memory is not None:
            proj = proj + memory
        # N x T x O
        out = self.out_proj(proj)
        if self.out_norm is not None:
            out = self.out_norm(out)
        # N x T x O
        return out, proj


class VariantRNN(nn.Module):
    """
    A custom rnn layer to support other features: -> RNN -> (Linear) -> (Norm) -> (NonLinear) -> (Dropout) ->
    """

    def __init__(self,
                 input_size: int,
                 rnn: str = "lstm",
                 norm: str = "",
                 hidden: int = 512,
                 project: int = -1,
                 non_linear: str = "relu",
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 add_forward_backward: bool = False):
        super(VariantRNN, self).__init__()
        if non_linear not in rnn_output_nonlinear:
            raise ValueError(f"Unsupported non_linear: {non_linear}")
        self.non_linear = rnn_output_nonlinear[non_linear]
        self.rnn = PyTorchRNN(rnn,
                              input_size,
                              hidden,
                              num_layers=1,
                              dropout=0,
                              bidirectional=bidirectional)
        self.add_forward_backward = add_forward_backward and bidirectional
        if bidirectional and not add_forward_backward:
            hidden *= 2
        self.proj = nn.Linear(hidden, project) if project > 0 else None
        self.norm = Normalize1d(
            norm, project if project > 0 else hidden) if norm else None
        self.drop = nn.Dropout(dropout) if dropout != 0 else None

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x Ti x O
        """
        out = var_len_rnn_forward(
            self.rnn,
            inp,
            inp_len=inp_len,
            enforce_sorted=False,
            add_forward_backward=self.add_forward_backward)
        # proj
        if self.proj:
            out = self.proj(out)
        # add ln
        if self.norm:
            out = self.norm(out)
        # nonlinear
        if self.non_linear:
            out = self.non_linear(out)
        # dropout
        if self.drop:
            out = self.drop(out)
        return out
