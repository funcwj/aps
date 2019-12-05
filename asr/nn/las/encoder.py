#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as tf

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


def encoder_instance(encoder_type, input_size, output_size, **kwargs):
    """
    Return encoder instance
    """
    supported_encoder = {
        "common": TorchEncoder,
        "custom": CustomEncoder,
        "tdnn": TdnnRnnEncoder
    }
    if encoder_type not in supported_encoder:
        raise RuntimeError(f"Unknown encoder type: {encoder_type}")
    return supported_encoder[encoder_type](input_size, output_size, **kwargs)


class TorchEncoder(nn.Module):
    """
    PyTorch's RNN encoder
    """
    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_size=512,
                 rnn_dropout=0.0,
                 rnn_bidir=False):
        super(TorchEncoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        self.rnns = supported_rnn[RNN](input_size,
                                       rnn_size,
                                       rnn_layers,
                                       batch_first=True,
                                       dropout=rnn_dropout,
                                       bidirectional=rnn_bidir)
        self.proj = nn.Linear(rnn_size if not rnn_bidir else rnn_size * 2,
                              output_size)

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(self, x_pad, x_len):
        """
        x_pad: (N) x Ti x F
        x_len: (N) x Ti
        """
        max_len = x_pad.size(1)
        if x_len is not None:
            x_pad = pack_padded_sequence(x_pad, x_len, batch_first=True)
        # extend dim when inference
        else:
            if x_pad.dim() not in [2, 3]:
                raise RuntimeError("RNN expect input dim as 2 or 3, " +
                                   f"got {x_pad.dim():d}")
            if x_pad.dim() != 3:
                x_pad = th.unsqueeze(x_pad, 0)
        y, _ = self.rnns(x_pad)
        # using unpacked sequence
        # y: NxTxD
        if x_len is not None:
            y, _ = pad_packed_sequence(y,
                                       batch_first=True,
                                       total_length=max_len)
        y = self.proj(y)
        return y, x_len


class CustomRnnLayer(nn.Module):
    """
    A custom rnn layer for PyramidEncoder
    """
    def __init__(self,
                 input_size,
                 hidden_size=512,
                 proj_size=None,
                 rnn="lstm",
                 layernorm=False,
                 dropout=0.0,
                 bidirectional=False,
                 add_forward_backward=False):
        super(CustomRnnLayer, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")

        self.rnn = supported_rnn[RNN](input_size,
                                      hidden_size,
                                      1,
                                      batch_first=True,
                                      bidirectional=bidirectional)
        self.add = add_forward_backward and bidirectional
        self.dpt = nn.Dropout(dropout) if dropout != 0 else None
        if not add_forward_backward and bidirectional:
            hidden_size *= 2
        self.lnm = nn.LayerNorm(hidden_size) if layernorm else None
        self.lin = nn.Linear(hidden_size, proj_size) if proj_size else None

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(self, x_pad, x_len):
        """
        x_pad: (N) x Ti x F
        x_len: (N) x Ti
        """
        max_len = x_pad.size(1)
        if x_len is not None:
            x_pad = pack_padded_sequence(x_pad, x_len, batch_first=True)
        # extend dim when inference
        else:
            if x_pad.dim() not in [2, 3]:
                raise RuntimeError("RNN expect input dim as 2 or 3, " +
                                   f"got {x_pad.dim()}")
            if x_pad.dim() != 3:
                x_pad = th.unsqueeze(x_pad, 0)
        y, _ = self.rnn(x_pad)
        # x: NxTxD
        if x_len is not None:
            y, _ = pad_packed_sequence(y,
                                       batch_first=True,
                                       total_length=max_len)
        # add forward & backward
        if self.add:
            f, b = th.chunk(y, 2, dim=-1)
            y = f + b
        # dropout
        if self.dpt:
            y = self.dpt(y)
        # add ln
        if self.lnm:
            y = self.lnm(y)
        # proj
        if self.lin:
            y = self.lin(y)
        return y


class CustomEncoder(nn.Module):
    """
    Customized RNN layer (egs: PyramidEncoder)
    """
    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_bidir=True,
                 rnn_dropout=0.0,
                 hidden_size=512,
                 proj_size=None,
                 layernorm=False,
                 use_pyramid=False,
                 add_forward_backward=False):
        super(CustomEncoder, self).__init__()

        def derive_in_size(layer_idx):
            """
            Compute input size of layer-i
            """
            if layer_idx == 0:
                in_size = input_size
            else:
                if proj_size:
                    return proj_size
                else:
                    in_size = hidden_size
                    if rnn_bidir and not add_forward_backward:
                        in_size = in_size * 2
                if use_pyramid:
                    in_size = in_size * 2
            return in_size

        rnn_list = []
        for i in range(rnn_layers):
            rnn_list.append(
                CustomRnnLayer(derive_in_size(i),
                               hidden_size=hidden_size,
                               rnn=rnn,
                               proj_size=proj_size
                               if i != rnn_layers - 1 else output_size,
                               layernorm=layernorm,
                               dropout=rnn_dropout,
                               bidirectional=rnn_bidir,
                               add_forward_backward=add_forward_backward))
        self.rnns = nn.ModuleList(rnn_list)
        self.use_pyramid = use_pyramid

    def flat(self):
        for layer in self.rnns:
            layer.flat()

    def _ds(self, x_pad, x_len):
        """
        Do downsampling for RNN output
        """
        _, T, _ = x_pad.shape
        # concat
        if T % 2:
            x_pad = x_pad[:, :-1]
        ctx = [x_pad[:, ::2], x_pad[:, 1::2]]
        x_pad = th.cat(ctx, -1)
        if x_len is not None:
            x_len = x_len // 2
        return x_pad, x_len

    def forward(self, x_pad, x_len):
        """
        args:
            x_pad: (N) x Ti x F
            x_len: (N) x Ti
        """
        for index, layer in enumerate(self.rnns):
            if index != 0 and self.use_pyramid:
                x_pad, x_len = self._ds(x_pad, x_len)
            x_pad = layer(x_pad, x_len)
        return x_pad, x_len


class TdnnLayer(nn.Module):
    """
    Implement a TDNN layer using conv1d operations
    """
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size=3,
                 steps=2,
                 dilat=1):
        super(TdnnLayer, self).__init__()
        self.conv1d = nn.Conv1d(input_size,
                                output_size,
                                kernel_size,
                                stride=steps,
                                dilation=dilat,
                                padding=(dilat * (kernel_size - 1)) // 2)
        self.norm = nn.BatchNorm1d(output_size)
        self.ds = steps

    def forward(self, x):
        """
        x: N x T x F
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                f"TdnnLayer accepts 2/3D tensor, got {x.dim()} instead")
        if x.dim() == 2:
            x = x[None, ...]
        # N x T x F => N x F x T
        x = x.transpose(1, 2)
        # conv & bn
        y = self.conv1d(x)
        y = y[..., :x.shape[-1] // self.ds]
        y = tf.relu(y)
        y = self.norm(y)
        # N x F x T => N x T x F
        y = y.transpose(1, 2)
        return y


class TdnnRnnEncoder(nn.Module):
    """
    Using TDNN (1D-conv) to downsample input sequences
    """
    def __init__(self,
                 input_size,
                 output_size,
                 tdnn_dim=512,
                 tdnn_layers=2,
                 tdnn_stride="2,2",
                 tdnn_dilats="1,1",
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_bidir=True,
                 rnn_dropout=0.2,
                 rnn_proj=None,
                 rnn_hidden=512):
        super(TdnnRnnEncoder, self).__init__()

        stride_conf = [int(t) for t in tdnn_stride.split(",")]
        dilats_conf = [int(t) for t in tdnn_dilats.split(",")]
        if len(stride_conf) != len(dilats_conf) or len(
                stride_conf) != tdnn_layers:
            raise RuntimeError("Errors in tdnn_stride/tdnn_dilats existed")
        tdnn_list = []
        for i in range(tdnn_layers):
            tdnn_list.append(
                TdnnLayer(input_size if i == 0 else tdnn_dim,
                          tdnn_dim,
                          kernel_size=3,
                          steps=stride_conf[i],
                          dilat=dilats_conf[i]))
        self.tdnn = nn.Sequential(*tdnn_list)
        self.rnns = CustomEncoder(tdnn_dim,
                                  output_size,
                                  rnn=rnn,
                                  rnn_layers=rnn_layers,
                                  rnn_bidir=rnn_bidir,
                                  rnn_dropout=rnn_dropout,
                                  proj_size=rnn_proj,
                                  hidden_size=rnn_hidden)

    def forward(self, x_pad, x_len):
        """
        x_pad: (N) x Ti x F
        x_len: (N) x Ti
        """
        if x_len is not None:
            div = 2**len(self.tdnn)
            x_len = x_len // div
        x_pad = self.tdnn(x_pad)
        return self.rnns(x_pad, x_len)
