#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence


def encoder_instance(encoder_type, input_size, output_size, **kwargs):
    """
    Return encoder instance
    """
    supported_encoder = {"common": TorchEncoder, "layernorm": LayerNormDecoder}
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
        self.rnn = supported_rnn[RNN](input_size,
                                      hidden_size,
                                      num_layers,
                                      batch_first=True,
                                      dropout=dropout,
                                      bidirectional=bidirectional)
        self.linear = nn.Linear(
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
        y, _ = self.rnn(x_pad)
        # using unpacked sequence
        # y: NxTxD
        if x_len is not None:
            y, _ = pad_packed_sequence(y,
                                       batch_first=True,
                                       total_length=max_len)
        y = self.linear(y)
        return y


class LnDecoderLayer(nn.Module):
    """
    A specific layer for LnDecoder
    """

    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 ln=True,
                 dropout=0.0,
                 bidirectional=False,
                 add_forward_backward=False):
        super(LnDecoderLayer, self).__init__()
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
        self.ln = nn.LayerNorm(output_size) if ln else None

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
        if self.ln:
            y = self.ln(y)
            y = F.relu(y)
        return y


class LayerNormDecoder(nn.Module):
    """
    Using layernorm between recurrent layers
    """

    def __init__(self,
                 input_size,
                 output_size,
                 rnn="lstm",
                 num_layers=3,
                 hidden_size=512,
                 dropout=0.0,
                 bidirectional=False):
        super(LayerNormDecoder, self).__init__()
        self.rnn = nn.ModuleList()
        for i in range(num_layers):
            self.rnn.append(
                LnDecoderLayer(input_size if i == 0 else hidden_size,
                               hidden_size,
                               rnn=rnn,
                               ln=i != num_layers - 1,
                               dropout=dropout if i != num_layers - 1 else 0,
                               bidirectional=bidirectional,
                               add_forward_backward=bidirectional))

        self.linear = nn.Linear(hidden_size, output_size)

    def flat(self):
        for c in self.rnn:
            c.flat()

    def forward(self, x_pad, x_len):
        """
        x_pad: (N) x Ti x F
        x_len: (N) x Ti
        """
        for layer in self.rnn:
            x_pad = layer(x_pad, x_len)
        y = self.linear(x_pad)
        return y


def foo():
    nnet = LayerNormDecoder(10,
                            20,
                            hidden_size=32,
                            dropout=0.4,
                            bidirectional=True)
    print(nnet)
    x = th.rand(30, 16, 10)
    y = nnet(x, None)
    print(y.shape)


if __name__ == "__main__":
    foo()