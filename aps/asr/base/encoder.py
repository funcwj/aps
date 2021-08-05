#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union, List, Dict

from aps.asr.base.component import VariantRNN, FSMN, Conv1d, Conv2d, PyTorchRNN
from aps.asr.base.component import rnn_output_nonlinear, var_len_rnn_forward
from aps.asr.base.jit import LSTM
from aps.libs import Register

BaseEncoder = Register("base_encoder")
EncRetType = Tuple[th.Tensor, Optional[th.Tensor]]


def encoder_instance(enc_type: str, inp_features: int, out_features: int,
                     enc_kwargs: Dict, enc_class: Dict) -> nn.Module:
    """
    Return encoder instance
    """

    def encoder(enc_type, inp_features, out_features: int, **kwargs):
        if enc_type not in enc_class:
            raise RuntimeError(f"Unknown encoder type: {enc_type}")
        enc_cls = enc_class[enc_type]
        return enc_cls(inp_features, out_features, **kwargs)

    if enc_type != "concat":
        return encoder(enc_type, inp_features, out_features, **enc_kwargs)
    else:
        enc_layers = []
        num_enc_layers = len(enc_kwargs)
        if num_enc_layers <= 1:
            raise ValueError(
                "Please use >=2 encoders for \'concat\' type encoder")
        for i, (name, kwargs) in enumerate(enc_kwargs.items()):
            if i != num_enc_layers - 1:
                enc_layer = encoder(
                    name,
                    inp_features if i == 0 else enc_layers[-1].out_features, -1,
                    **kwargs)
            else:
                enc_layer = encoder(name, enc_layers[-1].out_features,
                                    out_features, **kwargs)
            enc_layers.append(enc_layer)
        return ConcatEncoder(enc_layers)


class ConcatEncoder(nn.ModuleList):
    """
    Concatenation of the encoders (actually nn.ModuleList)
    """

    def __init__(self, enc_list: List[nn.Module]) -> None:
        super(ConcatEncoder, self).__init__(enc_list)

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        for encoder in self._modules.values():
            inp, inp_len = encoder(inp, inp_len)
        return inp, inp_len


class EncoderBase(nn.Module):
    """
    Add inp_features/out_features attributions to the encoder objects
    """

    def __init__(self, inp_features: int, out_features: int):
        super(EncoderBase, self).__init__()
        # NOTE: for out_features == -1, we will workout it automatically
        self.inp_features = inp_features
        self.out_features = out_features


@BaseEncoder.register("pytorch_rnn")
class PyTorchRNNEncoder(EncoderBase):
    """
    PyTorch's RNN encoder: (Linear) -> RNN -> (Linear) -> (NonLinear)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 input_proj: int = -1,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 hidden_proj: int = -1,
                 dropout: int = 0.2,
                 bidirectional: bool = False,
                 non_linear: str = "none"):
        super(PyTorchRNNEncoder, self).__init__(inp_features, out_features)
        if non_linear not in rnn_output_nonlinear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_proj > 0:
            self.proj = nn.Linear(inp_features, input_proj)
        else:
            self.proj = None
        self.rnns = PyTorchRNN(rnn,
                               input_proj if input_proj > 0 else inp_features,
                               hidden,
                               num_layers=num_layers,
                               dropout=dropout,
                               proj_size=hidden_proj,
                               bidirectional=bidirectional)
        hidden_size = hidden_proj if hidden_proj > 0 else hidden
        factor = 2 if bidirectional else 1
        if out_features > 0:
            self.outp = nn.Linear(hidden_size * factor, out_features)
            self.non_linear = rnn_output_nonlinear[non_linear]
        else:
            self.outp = None
            self.non_linear = None
            self.out_features = hidden_size * factor

    def flat(self):
        self.rnns.flatten_parameters()

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x T x F
            inp_len (Tensor): N x T
        Return:
            out (Tensor): N x T x F
            inp_len (Tensor): N x T
        """
        if self.proj is not None:
            inp = tf.relu(self.proj(inp))
        out = var_len_rnn_forward(self.rnns,
                                  inp,
                                  inp_len=inp_len,
                                  enforce_sorted=False,
                                  add_forward_backward=False)
        if self.outp is not None:
            out = self.outp(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out, inp_len


@BaseEncoder.register("jit_lstm")
class JitLSTMEncoder(EncoderBase):
    """
    LSTM encoder (see aps.asr.base.jit): (Linear) -> JitLSTM -> (Linear) -> (NonLinear)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 input_proj: int = -1,
                 num_layers: int = 3,
                 hidden: int = 512,
                 hidden_proj: int = None,
                 dropout: int = 0.2,
                 bidirectional: bool = False,
                 layer_norm: bool = False,
                 non_linear: str = "none"):
        super(JitLSTMEncoder, self).__init__(inp_features, out_features)
        if non_linear not in rnn_output_nonlinear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_proj > 0:
            self.proj = nn.Linear(inp_features, input_proj)
        else:
            self.proj = None
        self.rnns = LSTM(input_proj if input_proj > 0 else inp_features,
                         hidden,
                         dropout=dropout,
                         project=hidden_proj,
                         num_layers=num_layers,
                         layer_norm=layer_norm,
                         bidirectional=bidirectional)
        hidden_size = hidden_proj if hidden_proj > 0 else hidden
        factor = 2 if bidirectional else 1
        if out_features > 0:
            self.outp = nn.Linear(hidden_size * factor, out_features)
            self.non_linear = rnn_output_nonlinear[non_linear]
        else:
            self.outp = None
            self.non_linear = None
            self.out_features = hidden_size * factor

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x T x F
            inp_len (Tensor): N x T
        Return:
            out (Tensor): N x T x F
            inp_len (Tensor): N x T
        """
        if self.proj is not None:
            inp = tf.relu(self.proj(inp))
        out = self.rnns(inp)[0]
        if self.outp is not None:
            out = self.outp(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out, inp_len


@BaseEncoder.register("variant_rnn")
class VariantRNNEncoder(EncoderBase):
    """
    Stack of variant RNN layer: [ ... -> VariantRNN -> ... ]
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 rnn: str = "lstm",
                 hidden: int = 512,
                 num_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.0,
                 dropout_input: bool = True,
                 project: int = -1,
                 non_linear: str = "tanh",
                 norm: str = "",
                 pyramid_stack: bool = False,
                 add_forward_backward: bool = False):
        super(VariantRNNEncoder, self).__init__(inp_features, out_features)

        def derive_inp_size(layer_idx: int) -> int:
            """
            Compute input size of layer-i
            """
            if layer_idx == 0:
                in_size = inp_features
            else:
                if project > 0:
                    return project
                else:
                    in_size = hidden
                    if bidirectional and not add_forward_backward:
                        in_size = in_size * 2
                if pyramid_stack:
                    in_size = in_size * 2
            return in_size

        self.pyramid = pyramid_stack
        if bidirectional and not add_forward_backward:
            factor = 2
        else:
            factor = 1
        self.out_features = out_features if out_features > 0 else hidden * factor
        self.enc_layers = nn.ModuleList([
            VariantRNN(
                derive_inp_size(i),
                rnn=rnn,
                norm=norm if i != num_layers - 1 else "",
                hidden=hidden,
                project=project if i != num_layers - 1 else self.out_features,
                dropout=dropout if i != num_layers - 1 else 0,
                bidirectional=bidirectional,
                non_linear=non_linear if i != num_layers - 1 else "none",
                add_forward_backward=add_forward_backward)
            for i in range(num_layers)
        ])

    def _subsample_concat(self, inp: th.Tensor,
                          inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Do subsampling for RNN output
        """
        if inp.shape[1] % 2:
            inp = inp[:, :-1]
        # concat
        inp = th.cat([inp[:, ::2], inp[:, 1::2]], -1)
        return inp, None if inp_len is None else inp_len // 2

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): (N) x Ti x F
            inp_len (Tensor or None): (N) x Ti
        Return:
            out_pad (Tensor): (N) x To x F
            out_len (Tensor or None): (N) x To
        """
        for i, layer in enumerate(self.enc_layers):
            if i != 0 and self.pyramid:
                inp, inp_len = self._subsample_concat(inp, inp_len)
            inp = layer(inp, inp_len)
        return inp, inp_len


@BaseEncoder.register("conv1d")
class Conv1dEncoder(EncoderBase):
    """
    The stack of TDNN (conv1d) layers with optional time reduction
    """
    Conv1dParam = Union[List[int], int]

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 dim: int = 512,
                 norm: str = "BN",
                 num_layers: int = 3,
                 kernel: Conv1dParam = 3,
                 stride: Conv1dParam = 2,
                 dilation: Conv1dParam = 1,
                 dropout: float = 0,
                 for_streaming: bool = False):
        super(Conv1dEncoder, self).__init__(inp_features, out_features)

        def int2list(param, repeat):
            return [param] * repeat if isinstance(param, int) else param

        self.kernel = int2list(kernel, num_layers)
        self.stride = int2list(stride, num_layers)
        self.dilation = int2list(dilation, num_layers)
        self.out_features = out_features if out_features > 0 else dim
        self.enc_layers = nn.ModuleList([
            Conv1d(inp_features if i == 0 else dim,
                   dim if i != num_layers - 1 else self.out_features,
                   norm=norm,
                   kernel_size=self.kernel[i],
                   stride=self.stride[i],
                   dilation=self.dilation[i],
                   dropout=dropout,
                   for_streaming=for_streaming) for i in range(num_layers)
        ])
        self.out_features = dim

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x To x O
            out_len (Tensor or None)
        """
        for conv1d in self.enc_layers:
            inp = conv1d(inp)
            if inp_len is not None:
                inp_len = conv1d.compute_outp_dim(inp_len)
        return inp, inp_len


@BaseEncoder.register("conv2d")
class Conv2dEncoder(EncoderBase):
    """
    The stack of conv2d layers with optional time reduction
    """
    Conv2dParam = Union[List[int], int, List[Tuple[int]]]

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 channel: Union[int, List[int]] = 32,
                 in_channels: int = 1,
                 norm: str = "BN",
                 num_layers: int = 3,
                 kernel: Conv2dParam = 3,
                 stride: Conv2dParam = 2,
                 for_streaming: bool = False):
        super(Conv2dEncoder, self).__init__(inp_features, out_features)

        def param2need(param, num_layers):
            if isinstance(param, int):
                return [(param, param)] * num_layers
            else:
                if isinstance(param[0], int):
                    return [(p, p) for p in param]
                else:
                    return param

        self.kernel = param2need(kernel, num_layers)
        self.stride = param2need(stride, num_layers)
        if isinstance(channel, int):
            channel = [channel] * num_layers

        self.enc_layers = nn.ModuleList([
            Conv2d(in_channels if i == 0 else channel[i - 1],
                   channel[i],
                   kernel_size=self.kernel[i],
                   norm=norm,
                   stride=self.stride[i],
                   for_streaming=for_streaming) for i in range(num_layers)
        ])
        # freq axis
        freq_dim = inp_features
        for conv2d in self.enc_layers:
            freq_dim = conv2d.compute_outp_dim(freq_dim, 1)
        freq_x_channel = freq_dim * channel[-1]
        if out_features > 0:
            self.out_features = out_features
            self.outp = nn.Linear(freq_x_channel, out_features)
        else:
            self.out_features = freq_x_channel
            self.outp = None

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x T x F, input
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x F x O
            out_len (Tensor or None)
        """
        for conv2d in self.enc_layers:
            # N x C x T x F
            inp = conv2d(inp)
            if inp_len is not None:
                # time axis
                inp_len = conv2d.compute_outp_dim(inp_len, 0)
        N, C, T, F = inp.shape
        if C * F != self.out_features:
            raise ValueError(
                f"Got out_features = {C * F}, but self.out_features " +
                f"= {self.out_features} ")
        # N x T x C x F
        inp = inp.transpose(1, 2).contiguous()
        out = inp.view(N, T, -1)
        if self.outp is not None:
            out = self.outp(out)
        return out, inp_len


@BaseEncoder.register("fsmn")
class FSMNEncoder(EncoderBase):
    """
    Stack of FSMN layers, with optional residual connection
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 dim: int = 1024,
                 project: int = 512,
                 num_layers: int = 4,
                 residual: bool = True,
                 lctx: int = 3,
                 rctx: int = 3,
                 norm: str = "BN",
                 dilation: Union[List[int], int] = 1,
                 dropout: float = 0.0,
                 for_streaming: bool = False):
        super(FSMNEncoder, self).__init__(inp_features, out_features)
        if isinstance(dilation, int):
            dilation = [dilation] * num_layers
        self.enc_layers = nn.ModuleList([
            FSMN(inp_features if i == 0 else dim,
                 dim if i != num_layers - 1 else out_features,
                 project,
                 lctx=lctx,
                 rctx=rctx,
                 norm=norm if i != num_layers - 1 else "none",
                 dilation=dilation[i],
                 dropout=dropout,
                 for_streaming=for_streaming) for i in range(num_layers)
        ])
        self.residual = residual

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x T x F, input
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x T x F
            out_len (Tensor or None)
        """
        memory = th.jit.annotate(Optional[th.Tensor], None)
        for fsmn in self.enc_layers:
            if self.residual:
                inp, memory = fsmn(inp, memory=memory)
            else:
                inp, _ = fsmn(inp, memory=memory)
        return inp, inp_len
