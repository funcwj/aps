#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.encoder import PyTorchRNNEncoder, Conv1dEncoder, Conv2dEncoder, FSMNEncoder
from aps.asr.base.encoder import EncoderBase, EncRetType, rnn_output_nonlinear
from aps.streaming_asr.base.component import PyTorchNormLSTM
from aps.libs import Register

from typing import Union, List, Tuple, Optional

StreamingEncoder = Register("streaming_encoder")


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
        self.reset()

    def reset(self):
        self.hx = None

    def process(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        """
        if self.proj is not None:
            chunk = tf.relu(self.proj(chunk))
        chunk, self.hx = self.rnns(chunk, self.hx)
        out = self.outp(chunk)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Used for training and offline evaluation
        """
        self.reset()
        out = self.process(inp)
        return out, inp_len


@StreamingEncoder.register("fsmn")
class StreamingFSMNEncoder(FSMNEncoder):
    """
    A streaming version of FSMN encoder
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
                 dropout: float = 0):
        super(StreamingFSMNEncoder, self).__init__(inp_features,
                                                   out_features,
                                                   dim=dim,
                                                   project=project,
                                                   num_layers=num_layers,
                                                   residual=residual,
                                                   lctx=lctx,
                                                   rctx=rctx,
                                                   norm=norm,
                                                   dilation=dilation,
                                                   dropout=dropout)


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
        self.reset()

    def reset(self):
        self.hx = [None] * len(self.lstm)

    def process(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        """
        if self.proj is not None:
            chunk = tf.relu(self.proj(chunk))

        hx = []
        for i, layer in enumerate(self.lstm):
            chunk, h = layer(chunk, hx=self.hx[i])
            hx.append(h)
        self.hx = hx

        out = self.outp(chunk)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Used for training and offline evaluation
        """
        self.reset()
        out = self.process(inp)
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

    def process(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        """
        for conv1d in self.enc_layers:
            chunk = conv1d(chunk)
        return chunk


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
                 stride: Conv2dParam = 2):
        super(StreamingConv2dEncoder, self).__init__(inp_features,
                                                     out_features,
                                                     channel=channel,
                                                     in_channels=in_channels,
                                                     norm=norm,
                                                     num_layers=num_layers,
                                                     kernel=kernel,
                                                     stride=stride,
                                                     for_streaming=True)

    def process(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        """
        for conv2d in self.enc_layers:
            chunk = conv2d(chunk)
        N, C, T, F = chunk.shape
        assert C * F == self.out_features
        # N x C x T x F => N x T x C x F
        out = chunk.transpose(1, 2).contiguous()
        return out.view(N, T, -1)