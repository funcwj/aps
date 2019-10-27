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
    supported_encoder = {"common": TorchEncoder, "pyramid": PyramidEncoder}
    if encoder_type not in supported_encoder:
        raise RuntimeError("Unknown encoder type: {}".format(encoder_type))
    return supported_encoder[encoder_type](input_size, output_size, **kwargs)


class TorchEncoder(nn.Module):
    """
    PyTorch's RNN encoder
    """
    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0,
                 bidirectional=False):
        super(TorchEncoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError("unknown RNN type: {}".format(RNN))
        self.rnns = supported_rnn[RNN](input_size,
                                       hidden_size,
                                       num_layers,
                                       batch_first=True,
                                       dropout=dropout,
                                       bidirectional=bidirectional)
        self.proj = nn.Linear(
            hidden_size if not bidirectional else hidden_size * 2, output_size)

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
                raise RuntimeError("RNN expect input dim as 2 or 3, "
                                   "got {:d}".format(x_pad.dim()))
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
                 output_size,
                 rnn="lstm",
                 ln=True,
                 dropout=0.0,
                 bidirectional=False,
                 add_forward_backward=False):
        super(CustomRnnLayer, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        if RNN not in supported_rnn:
            raise RuntimeError("unknown RNN type: {}".format(RNN))

        self.rnn = supported_rnn[RNN](input_size,
                                      output_size,
                                      1,
                                      batch_first=True,
                                      bidirectional=bidirectional)
        self.add = add_forward_backward and bidirectional
        self.dpt = nn.Dropout(dropout) if dropout != 0 else None
        self.lnm = nn.LayerNorm(output_size if self.add else output_size *
                                2) if ln else None

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
        return y


class PyramidEncoder(nn.Module):
    """
    PyramidEncoder using subsampling in RNN structure
    """
    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0,
                 pyramid=False,
                 ln=False,
                 bidirectional=False,
                 add_forward_backward=False):
        super(PyramidEncoder, self).__init__()

        def derive_in_size(layer_idx):
            """
            Compute input size of layer-x
            """
            if layer_idx == 0:
                in_size = input_size
            else:
                in_size = hidden_size
                if bidirectional and not add_forward_backward:
                    in_size = in_size * 2
                if pyramid:
                    in_size = in_size * 2
            return in_size

        rnn_layers = []
        for i in range(num_layers):
            rnn_layers.append(
                CustomRnnLayer(derive_in_size(i),
                               hidden_size,
                               rnn=rnn,
                               ln=(i != num_layers - 1 and ln),
                               dropout=dropout if i != num_layers - 1 else 0,
                               bidirectional=bidirectional,
                               add_forward_backward=bidirectional
                               and add_forward_backward))
        self.rnns = nn.ModuleList(rnn_layers)
        self.proj = nn.Linear(derive_in_size(num_layers), output_size)
        self.pyramid = pyramid

    def flat(self):
        for layer in self.rnns:
            layer.flat()

    def forward(self, x_pad, x_len):
        """
        x_pad: (N) x Ti x F
        x_len: (N) x Ti
        """
        for layer in self.rnns:
            x_pad = layer(x_pad, x_len)
            if self.pyramid:
                _, T, F = x_pad.shape
                if T % 2:
                    x_pad = x_pad[:, :-1]
                # concat
                x_pad = x_pad.reshape(-1, T // 2, 2 * F)
                if x_len is not None:
                    x_len = x_len // 2
        x_pad = self.proj(x_pad)
        return x_pad, x_len


def foo():
    nnet = PyramidEncoder(10,
                          20,
                          hidden_size=32,
                          num_layers=3,
                          dropout=0.4,
                          pyramid=True,
                          ln=True,
                          bidirectional=True,
                          add_forward_backward=False)
    print(nnet)
    x = th.rand(30, 32, 10)
    y, z = nnet(x, None)
    print(y.shape)


if __name__ == "__main__":
    foo()