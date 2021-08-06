#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.base.encoder import PyTorchRNNEncoder, Conv1dEncoder, Conv2dEncoder, FSMNEncoder
from aps.asr.base.encoder import EncRetType
from aps.libs import Register

from typing import Union, List, Tuple, Optional

StreamingEncoder = Register("streaming_encoder")


@StreamingEncoder.register("pytorch_rnn")
class StreamingRNNEncoder(PyTorchRNNEncoder):
    """
    A streaming version of RNN encoder
    """
    # NOTE: for LSTM only. Change to Optional[th.Tensor] for RNN/GRU
    hx: Optional[Tuple[th.Tensor, th.Tensor]]

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 input_proj: int = -1,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 hidden_proj: int = -1,
                 dropout: float = 0.0,
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

    @th.jit.export
    def reset(self):
        self.hx = None

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk per step (for online processing)
        Args:
            chunk (Tensor): N x (T) x D
        Return:
            chunk (Tensor): N x (T) x D
        """
        if chunk.dim() == 2:
            chunk = chunk[:, None]
        if self.proj is not None:
            chunk = tf.relu(self.proj(chunk))
        out, hx = self.rnns(chunk, self.hx)
        self.hx = hx
        if self.outp is not None:
            out = self.outp(out)
        if self.non_linear is not None:
            out = self.non_linear(out)
        return out

    def forward(self, inp: th.Tensor,
                inp_len: Optional[th.Tensor]) -> EncRetType:
        """
        Used for training and offline evaluation
        """
        self.reset()
        out = self.step(inp)
        return out, inp_len


@StreamingEncoder.register("fsmn")
class StreamingFSMNEncoder(FSMNEncoder):
    """
    A streaming version of FSMN encoder (stride = 1)
    """
    FSMNParam = Union[List[int], int]
    hc: List[th.Tensor]
    hm: List[th.Tensor]

    def __init__(self,
                 inp_features: int,
                 out_features: int,
                 dim: int = 1024,
                 project: int = 512,
                 num_layers: int = 4,
                 lctx: FSMNParam = 3,
                 rctx: FSMNParam = 3,
                 residual: bool = False,
                 norm: str = "BN",
                 dropout: float = 0.0):
        super(StreamingFSMNEncoder, self).__init__(inp_features,
                                                   out_features,
                                                   dim=dim,
                                                   project=project,
                                                   num_layers=num_layers,
                                                   residual=residual,
                                                   lctx=lctx,
                                                   rctx=rctx,
                                                   norm=norm,
                                                   dilation=1,
                                                   dropout=dropout,
                                                   for_streaming=True)
        self.reset()

    @th.jit.export
    def reset(self):
        self.hc = [th.tensor(0)]
        self.hm = [th.tensor(0)]
        self.init = True

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        Args:
            chunk (Tensor): N x (T) x D
        Return:
            chunk (Tensor): N x (T) x D
        """
        hc, hm = [], []
        if not self.init:
            chunk = chunk[:, None]

        mem = th.tensor(0)
        for i, fsmn in enumerate(self.enc_layers):
            if not self.init:
                chunk = th.cat([self.hc[i], chunk], 1)
            hc.append(chunk[:, -self.ctx[i]:])
            if self.res:
                if i == 0:
                    chunk, mem = fsmn(chunk)
                else:
                    if not self.init:
                        mem = th.cat([self.hm[i - 1], mem], 1)
                    hm.append(mem[:, -self.ctx[i]:])
                    chunk, mem = fsmn(chunk, memory=mem)
            else:
                chunk = fsmn(chunk)[0]

        self.hc = hc
        self.hm = hm
        self.init = False
        return chunk


@StreamingEncoder.register("conv1d")
class StreamingConv1dEncoder(Conv1dEncoder):
    """
    A streaming version of conv1d encoder (prefer stride != 1)
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

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        Args:
            chunk (Tensor): N x T x D
        Return:
            chunk (Tensor): N x T x D
        """
        for conv1d in self.enc_layers:
            chunk = conv1d(chunk)
        return chunk


@StreamingEncoder.register("conv2d")
class StreamingConv2dEncoder(Conv2dEncoder):
    """
    A streaming version of conv2d encoder (prefer stride != 1)
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

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Process one chunk (for online processing)
        Args:
            chunk (Tensor): N x T x D
        Return:
            chunk (Tensor): N x T x D
        """
        for conv2d in self.enc_layers:
            chunk = conv2d(chunk)
        N, _, T, _ = chunk.shape
        # N x C x T x F => N x T x C x F
        out = chunk.transpose(1, 2).contiguous()
        out = out.view(N, T, -1)
        if self.outp is not None:
            out = self.outp(out)
        return out