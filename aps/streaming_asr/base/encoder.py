#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.encoder import PyTorchRNNEncoder, Conv1dEncoder, Conv2dEncoder, ConcatEncoder
from aps.asr.base.encoder import EncoderBase, EncRetType, rnn_output_nonlinear
from aps.streaming_asr.base.component import PyTorchNormLSTM
from aps.libs import Register

from typing import Dict, Union, List, Tuple, Optional

StreamingEncoder = Register("streaming_encoder")


def encoder_instance(enc_type: str, inp_features: int, out_features: int,
                     enc_kwargs: Dict) -> nn.Module:
    """
    Return streaming encoder instance
    """

    def encoder(enc_type, inp_features, **kwargs):
        if enc_type not in StreamingEncoder:
            raise RuntimeError(f"Unknown encoder type: {enc_type}")
        enc_cls = StreamingEncoder[enc_type]
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


@StreamingEncoder.register("pytorch_rnn")
class StreamingRNNEncoder(PyTorchRNNEncoder):
    """
    A streaming version of RNN encoder
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
                 non_linear: str = "none"):
        super(StreamingRNNEncoder, self).__init__(inp_features,
                                                  out_features,
                                                  input_proj=input_proj,
                                                  rnn=rnn,
                                                  num_layers=num_layers,
                                                  hidden=hidden,
                                                  hidden_proj=hidden_proj,
                                                  dropout=dropout,
                                                  bidirectional=False,
                                                  non_linear=non_linear)

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
        out = self.outp(out)
        # pass through non-linear
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out, inp_len


@StreamingEncoder.register("pytorch_lstm_ln")
class StreamingLSTMNormEncoder(EncoderBase):
    """
    A streaming version of LSTM + layer normalization encoder 
    """

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 input_proj: int = -1,
                 num_layers: int = 3,
                 hidden: int = 512,
                 hidden_proj: int = -1,
                 dropout: int = 0.2,
                 non_linear: str = "none"):
        super(StreamingLSTMNormEncoder, self).__init__(inp_features,
                                                       out_features)
        if non_linear not in rnn_output_nonlinear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_proj > 0:
            self.proj = nn.Linear(inp_features, input_proj)
            inp_features = input_proj
        else:
            self.proj = None
        self.lstm = nn.ModuleList([
            PyTorchNormLSTM(inp_features,
                            hidden_size=hidden,
                            proj_size=hidden_proj,
                            non_linear="relu",
                            dropout=dropout) for _ in range(num_layers)
        ])
        self.outp = nn.Linear(hidden_proj if hidden_proj > 0 else hidden,
                              out_features)
        self.non_linear = rnn_output_nonlinear[non_linear]

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
        for layer in self.lstm:
            inp = layer(inp)[0]
        out = self.outp(inp)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out, inp_len


@StreamingEncoder.register("conv1d")
class StreamingConv1dEncoder(Conv1dEncoder):
    """
    A streaming version of conv1d encoder
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
                 dropout: float = 0):
        super(StreamingConv1dEncoder, self).__init__(inp_features,
                                                     out_features,
                                                     dim=dim,
                                                     norm=norm,
                                                     num_layers=num_layers,
                                                     kernel=kernel,
                                                     stride=stride,
                                                     dilation=dilation,
                                                     dropout=dropout,
                                                     for_streaming=True)


@StreamingEncoder.register("conv2d")
class StreamingConv2dEncoder(Conv2dEncoder):
    """
    A streaming version of conv2d encoder
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
                 padding: Conv2dParam = 0):
        super(StreamingConv2dEncoder, self).__init__(inp_features,
                                                     out_features,
                                                     channel=channel,
                                                     in_channels=in_channels,
                                                     norm=norm,
                                                     num_layers=num_layers,
                                                     kernel=kernel,
                                                     stride=stride,
                                                     padding=padding,
                                                     for_streaming=True)