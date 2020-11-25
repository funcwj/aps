#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Tuple, Union, List, Dict
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from aps.asr.base.layers import VariantRNN, FSMN, TDNN


def encoder_instance(enc_type: str, inp_features: int, out_features: int,
                     enc_kwargs: Dict) -> nn.Module:
    """
    Return encoder instance
    """
    supported_encoder = {
        "vanilla_rnn": VanillaRNNEncoder,
        "variant_rnn": VariantRNNEncoder,
        "tdnn": TDNNEncoder,
        "fsmn": FSMNEncoder
    }

    def encoder(enc_type, inp_features, **kwargs):
        if enc_type not in supported_encoder:
            raise RuntimeError(f"Unknown encoder type: {enc_type}")
        enc_cls = supported_encoder[enc_type]
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
                f"Please use >=2 encoders for \'concat\' type encoder")
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

    def forward(
            self, inp: th.Tensor, inp_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
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


class VanillaRNNEncoder(EncoderBase):
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
        super(VanillaRNNEncoder, self).__init__(inp_features, out_features)
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        supported_non_linear = {
            "relu": tf.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh,
            "": None
        }
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if non_linear not in supported_non_linear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_project:
            self.proj = nn.Linear(inp_features, input_project)
        else:
            self.proj = None
        self.rnns = supported_rnn[RNN](
            inp_features if input_project is None else input_project,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional)
        self.outp = nn.Linear(hidden if not bidirectional else hidden * 2,
                              out_features)
        self.non_linear = supported_non_linear[non_linear]

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(
            self,
            inp: th.Tensor,
            inp_len: Optional[th.Tensor],
            max_len: Optional[int] = None
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            inp (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        Return:
            out (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        """
        self.rnns.flatten_parameters()
        if inp_len is not None:
            inp = pack_padded_sequence(inp,
                                       inp_len,
                                       batch_first=True,
                                       enforce_sorted=False)
        # extend dim when inference
        else:
            if inp.dim() not in [2, 3]:
                raise RuntimeError("VanillaRNNEncoder expects 2/3D Tensor, " +
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


class VariantRNNEncoder(EncoderBase):
    """
    Variant RNN layer (e.g., with pyramid stack, layernrom, projection layer, .etc)
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 bidirectional: bool = True,
                 dropout: float = 0.0,
                 hidden: int = 512,
                 project: Optional[int] = None,
                 layernorm: bool = False,
                 use_pyramid: bool = False,
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
                if use_pyramid:
                    in_size = in_size * 2
            return in_size

        self.rnns = nn.ModuleList([
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
        self.use_pyramid = use_pyramid

    def _subsample_concat(
            self, inp: th.Tensor, inp_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Do subsampling for RNN output
        """
        _, T, _ = inp.shape
        # concat
        if T % 2:
            inp = inp[:, :-1]
        ctx = [inp[:, ::2], inp[:, 1::2]]
        inp = th.cat(ctx, -1)
        if inp_len is not None:
            inp_len = inp_len // 2
        return inp, inp_len

    def forward(
            self, inp: th.Tensor, inp_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            inp (Tensor): (N) x Ti x F
            inp_len (Tensor or None): (N) x Ti
        Return:
            out_pad (Tensor): (N) x To x F
            out_len (Tensor or None): (N) x To
        """
        for i, layer in enumerate(self.rnns):
            if i != 0 and self.use_pyramid:
                inp, inp_len = self._subsample_concat(inp, inp_len)
            inp = layer(inp, inp_len)
        return inp, inp_len


class TDNNEncoder(EncoderBase):
    """
    The stack of TDNN (conv1d) layers with optional time reduction
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 dim: int = 512,
                 norm: str = "BN",
                 num_layers: int = 3,
                 stride: Union[List[int], int] = 2,
                 dilation: Union[List[int], int] = 1,
                 dropout: float = 0):
        super(TDNNEncoder, self).__init__(inp_features, out_features)
        if isinstance(stride, int):
            stride = [stride] * num_layers
        if isinstance(dilation, int):
            dilation = [dilation] * num_layers
        self.encs = nn.Sequential(*[
            TDNN(inp_features if i == 0 else dim,
                 dim if i != num_layers - 1 else out_features,
                 kernel_size=3,
                 norm=norm,
                 stride=stride[i],
                 dilation=dilation[i],
                 dropout=dropout,
                 subsampling=True) for i in range(num_layers)
        ])
        self.subsampling_factor = np.prod(stride)

    def forward(
            self, inp: th.Tensor, inp_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x To x O
            out_len (Tensor or None)
        """
        out = self.encs(inp)
        if inp_len is None:
            return out, None
        else:
            return out, inp_len // self.subsampling_factor


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
        self.layers = nn.ModuleList([
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

    def forward(
            self, inp: th.Tensor, inp_len: Optional[th.Tensor]
    ) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            inp (Tensor): N x T x F, input
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x F x O
            out_len (Tensor or None)
        """
        if inp.dim() not in [2, 3]:
            raise RuntimeError(
                f"FSMNEncoder expect 2/3D input, got {inp.dim()}")
        if inp.dim() == 2:
            inp = inp[None, ...]
        memory = None
        for fsmn in self.layers:
            if self.residual:
                inp, memory = fsmn(inp, memory=memory)
            else:
                inp, _ = fsmn(inp, memory=memory)
        inp = inp.transpose(1, 2)
        return inp, inp_len
