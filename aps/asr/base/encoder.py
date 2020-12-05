#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union, List, Dict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from aps.asr.base.layers import VariantRNN, FSMN, Conv1d, Conv2d, PyTorchRNN
from aps.libs import Register

AsrEncoder = Register("asr_encoder")
EncRetType = Tuple[th.Tensor, Optional[th.Tensor]]


def encoder_instance(enc_type: str, inp_features: int, out_features: int,
                     enc_kwargs: Dict) -> nn.Module:
    """
    Return encoder instance
    """

    def encoder(enc_type, inp_features, **kwargs):
        if enc_type not in AsrEncoder:
            raise RuntimeError(f"Unknown encoder type: {enc_type}")
        enc_cls = AsrEncoder[enc_type]
        return enc_cls(inp_features, **kwargs)

    if enc_type != "concat":
        return encoder(enc_type,
                       inp_features,
                       out_features=out_features,
                       **enc_kwargs)
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
                    inp_features if i == 0 else enc_layers[-1].out_features,
                    **kwargs)
            else:
                enc_layer = encoder(name,
                                    enc_layers[-1].out_features,
                                    out_features=out_features,
                                    **kwargs)
            enc_layers.append(enc_layer)
        return ConcatEncoder(enc_layers)


class ConcatEncoder(nn.Module):
    """
    Concatenation of the encoders
    """

    def __init__(self, enc_list: List[nn.Module]) -> None:
        super(ConcatEncoder, self).__init__()
        self.enc_list = nn.ModuleList(enc_list)

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x To x O
            out_len (Tensor or None): N
        """
        for enc in self.enc_list:
            inp, inp_len = enc(inp, inp_len)
        return inp, inp_len


class EncoderBase(nn.Module):
    """
    Base class for encoders
    """

    def __init__(self, inp_features, out_features):
        super(EncoderBase, self).__init__()
        self.inp_features = inp_features
        self.out_features = out_features


@AsrEncoder.register("pytorch_rnn")
class PyTorchRNNEncoder(EncoderBase):
    """
    PyTorch's RNN encoder
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 input_project: Optional[int] = None,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 dropout: int = 0.2,
                 bidirectional: bool = False,
                 non_linear: str = ""):
        super(PyTorchRNNEncoder, self).__init__(inp_features, out_features)
        supported_non_linear = {
            "relu": tf.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh,
            "": None
        }
        if non_linear not in supported_non_linear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_project:
            self.proj = nn.Linear(inp_features, input_project)
        else:
            self.proj = None
        self.rnns = PyTorchRNN(
            rnn,
            inp_features if input_project is None else input_project,
            hidden,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional)
        self.outp = nn.Linear(hidden if not bidirectional else hidden * 2,
                              out_features)
        self.non_linear = supported_non_linear[non_linear]

    def flat(self):
        self.rnns.flatten_parameters()

    def forward(self,
                inp: th.Tensor,
                inp_len: Optional[th.Tensor],
                max_len: Optional[int] = None) -> EncRetType:
        """
        Args:
            inp (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        Return:
            out (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        """
        self.flat()
        if inp_len is not None:
            inp = pack_padded_sequence(inp,
                                       inp_len,
                                       batch_first=True,
                                       enforce_sorted=False)
        else:
            if inp.dim() not in [2, 3]:
                raise RuntimeError("PyTorchRNNEncoder expects 2/3D Tensor, " +
                                   f"got {inp.dim():d}")
            if inp.dim() != 3:
                inp = th.unsqueeze(inp, 0)
        if self.proj:
            inp = tf.relu(self.proj(inp))
        rnn_out, _ = self.rnns(inp)
        # using unpacked sequence
        # rnn_out: N x T x D
        if inp_len is not None:
            rnn_out, _ = pad_packed_sequence(rnn_out,
                                             batch_first=True,
                                             total_length=max_len)
        out = self.outp(rnn_out)
        # pass through non-linear
        if self.non_linear:
            out = self.non_linear(out)
        return out, inp_len


@AsrEncoder.register("variant_rnn")
class VariantRNNEncoder(EncoderBase):
    """
    Variant RNN layer (e.g., with pyramid style, layernrom, projection, .etc)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 rnn: str = "lstm",
                 hidden: int = 512,
                 num_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.0,
                 project: Optional[int] = None,
                 layernorm: bool = False,
                 pyramid_stack: bool = False,
                 add_forward_backward: bool = False):
        super(VariantRNNEncoder, self).__init__(inp_features, out_features)

        def derive_in_size(layer_idx):
            """
            Compute input size of layer-i
            """
            if layer_idx == 0:
                in_size = inp_features
            else:
                if project:
                    return project
                else:
                    in_size = hidden
                    if bidirectional and not add_forward_backward:
                        in_size = in_size * 2
                if pyramid_stack:
                    in_size = in_size * 2
            return in_size

        self.enc_layers = nn.ModuleList([
            VariantRNN(
                derive_in_size(i),
                hidden_size=hidden,
                rnn=rnn,
                project_size=project if i != num_layers - 1 else out_features,
                layernorm=layernorm,
                dropout=dropout,
                bidirectional=bidirectional,
                add_forward_backward=add_forward_backward)
            for i in range(num_layers)
        ])
        self.pyramid_stack = pyramid_stack

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
            if i != 0 and self.pyramid_stack:
                inp, inp_len = self._subsample_concat(inp, inp_len)
            inp = layer(inp, inp_len)
        return inp, inp_len


@AsrEncoder.register("conv1d")
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
                 stride: Conv1dParam = 2,
                 dilation: Conv1dParam = 1,
                 dropout: float = 0):
        super(Conv1dEncoder, self).__init__(inp_features, out_features)

        def int2list(param, repeat):
            return [param] * repeat if isinstance(param, int) else param

        stride = int2list(stride, num_layers)
        dilation = int2list(dilation, num_layers)
        self.enc_layers = nn.ModuleList([
            Conv1d(inp_features if i == 0 else dim,
                   dim if i != num_layers - 1 else out_features,
                   kernel_size=3,
                   norm=norm,
                   stride=stride[i],
                   dilation=dilation[i],
                   dropout=dropout) for i in range(num_layers)
        ])

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
        for enc_layer in self.enc_layers:
            inp = enc_layer(inp)
            if inp_len is not None:
                inp_len = enc_layer.compute_outp_dim(inp_len)
        return inp, inp_len


@AsrEncoder.register("conv2d")
class Conv2dEncoder(EncoderBase):
    """
    The stack of conv2d layers with optional time reduction
    """
    Conv2dParam = Union[List[int], int, List[Tuple[int]]]

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 channel: Union[int, List[int]] = 32,
                 num_layers: int = 3,
                 kernel_size: Conv2dParam = 3,
                 padding: Conv2dParam = 0,
                 stride: Conv2dParam = 2):
        super(Conv2dEncoder, self).__init__(inp_features, out_features)

        def param2need(param, num_layers):
            if isinstance(param, int):
                return [(param, param)] * num_layers
            else:
                if isinstance(param[0], int):
                    return [(p, p) for p in param]
                else:
                    return param

        if isinstance(channel, int):
            channel = [channel] * num_layers
        kernel_size = param2need(kernel_size, num_layers)
        padding = param2need(padding, num_layers)
        stride = param2need(stride, num_layers)

        self.enc_layers = nn.ModuleList([
            Conv2d(1 if i == 0 else channel[i - 1],
                   channel[i],
                   kernel_size=kernel_size[i],
                   stride=stride[i],
                   padding=padding[i]) for i in range(num_layers)
        ])

        freq_dim = inp_features
        for enc_layer in self.enc_layers:
            # freq axis
            freq_dim = enc_layer.compute_outp_dim(freq_dim, 1)
        self.out_features = freq_dim * channel[-1]

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
        for enc_layer in self.enc_layers:
            # N x C x T x F
            inp = enc_layer(inp)
            if inp_len is not None:
                # time axis
                inp_len = enc_layer.compute_outp_dim(inp_len, 0)
        N, C, T, F = inp.shape
        if C * F != self.out_features:
            raise ValueError(
                f"Got out_features = {C * F}, but self.out_features " +
                f"= {self.out_features} ")
        # N x T x C x F
        inp = inp.transpose(1, 2).contiguous()
        inp = inp.view(N, T, -1)
        return inp, inp_len


@AsrEncoder.register("fsmn")
class FSMNEncoder(EncoderBase):
    """
    Stack of FSMN layers, with optional residual connection
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 project: int = 512,
                 num_layers: int = 4,
                 residual: bool = True,
                 lctx: int = 3,
                 rctx: int = 3,
                 norm: str = "BN",
                 dilation: Union[List[int], int] = 1,
                 dropout: float = 0):
        super(FSMNEncoder, self).__init__(inp_features, out_features)
        if isinstance(dilation, int):
            dilation = [dilation] * num_layers
        self.enc_layers = nn.ModuleList([
            FSMN(inp_features if i == 0 else out_features,
                 out_features,
                 project,
                 lctx=lctx,
                 rctx=rctx,
                 norm="" if i == num_layers - 1 else norm,
                 dilation=dilation[i],
                 dropout=dropout) for i in range(num_layers)
        ])
        self.residual = residual

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
        memory = None
        for fsmn in self.enc_layers:
            if self.residual:
                inp, memory = fsmn(inp, memory=memory)
            else:
                inp, _ = fsmn(inp, memory=memory)
        inp = inp.transpose(1, 2)
        return inp, inp_len
