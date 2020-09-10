#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as tf

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from aps.asr.base.layers import CustomRNNLayer, FSMNLayer, TDNNLayer


def encoder_instance(encoder_type, input_size, output_size, **kwargs):
    """
    Return encoder instance
    """
    supported_encoder = {
        "common_rnn": TorchRNNEncoder,
        "custom_rnn": CustomRNNEncoder,
        "tdnn": TDNNEncoder,
        "fsmn": FSMNEncoder,
        "tdnn_rnn": TimeDelayRNNEncoder,
        "tdnn_fsmn": TimeDelayFSMNEncoder
    }
    if encoder_type not in supported_encoder:
        raise RuntimeError(f"Unknown encoder type: {encoder_type}")
    return supported_encoder[encoder_type](input_size, output_size, **kwargs)


class TorchRNNEncoder(nn.Module):
    """
    PyTorch's RNN encoder
    """
    def __init__(self,
                 input_size,
                 output_size,
                 input_project=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.2,
                 rnn_bidir=False,
                 non_linear=""):
        super(TorchRNNEncoder, self).__init__()
        RNN = rnn.upper()
        supported_rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU, "RNN": nn.RNN}
        support_non_linear = {
            "relu": tf.relu,
            "sigmoid": th.sigmoid,
            "tanh": th.tanh,
            "": None
        }
        if RNN not in supported_rnn:
            raise RuntimeError(f"Unknown RNN type: {RNN}")
        if non_linear not in support_non_linear:
            raise ValueError(
                f"Unsupported output non-linear function: {non_linear}")
        if input_project:
            self.proj = nn.Linear(input_size, input_project)
        else:
            self.proj = None
        self.rnns = supported_rnn[RNN](
            input_size if input_project is None else input_project,
            rnn_hidden,
            rnn_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=rnn_bidir)
        self.outp = nn.Linear(rnn_hidden if not rnn_bidir else rnn_hidden * 2,
                              output_size)
        self.non_linear = support_non_linear[non_linear]

    def flat(self):
        self.rnn.flatten_parameters()

    def forward(self, inp, inp_len, max_len=None):
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
                raise RuntimeError("TorchRNNEncoder expects 2/3D Tensor, " +
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


class CustomRNNEncoder(nn.Module):
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
                 rnn_hidden=512,
                 rnn_project=None,
                 layernorm=False,
                 use_pyramid=False,
                 add_forward_backward=False):
        super(CustomRNNEncoder, self).__init__()

        def derive_in_size(layer_idx):
            """
            Compute input size of layer-i
            """
            if layer_idx == 0:
                in_size = input_size
            else:
                if rnn_project:
                    return rnn_project
                else:
                    in_size = rnn_hidden
                    if rnn_bidir and not add_forward_backward:
                        in_size = in_size * 2
                if use_pyramid:
                    in_size = in_size * 2
            return in_size

        rnn_list = []
        for i in range(rnn_layers):
            rnn_list.append(
                CustomRNNLayer(derive_in_size(i),
                               hidden_size=rnn_hidden,
                               rnn=rnn,
                               project_size=rnn_project
                               if i != rnn_layers - 1 else output_size,
                               layernorm=layernorm,
                               dropout=rnn_dropout,
                               bidirectional=rnn_bidir,
                               add_forward_backward=add_forward_backward))
        self.rnns = nn.ModuleList(rnn_list)
        self.use_pyramid = use_pyramid

    def _downsample_concat(self, inp, inp_len):
        """
        Do downsampling for RNN output
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

    def forward(self, inp, inp_len):
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
                inp, inp_len = self._downsample_concat(inp, inp_len)
            inp = layer(inp, inp_len)
        return inp, inp_len


def parse_str_int(str_or_int, num_layers):
    """
    Parse string or int, egs:
        1,1,2 => [1, 1, 2]
        2     => [2, 2, 2]
    """
    if isinstance(str_or_int, str):
        values = [int(t) for t in str_or_int.split(",")]
        if len(values) != num_layers:
            raise ValueError(f"Number of the layers: {num_layers} " +
                             f"do not match {str_or_int}")
    else:
        values = [str_or_int] * num_layers
    return values


class TDNNEncoder(nn.Module):
    """
    Stack of TDNNLayers
    """
    def __init__(self,
                 input_size,
                 output_size,
                 dim=512,
                 norm="BN",
                 num_layers=3,
                 stride="2,2,2",
                 dilation="1,1,1",
                 dropout=0):
        super(TDNNEncoder, self).__init__()
        stride_conf = parse_str_int(stride, num_layers)
        dilation_conf = parse_str_int(dilation, num_layers)
        tdnns = [
            TDNNLayer(input_size if i == 0 else dim,
                      dim,
                      kernel_size=3,
                      norm=norm,
                      stride=stride_conf[i],
                      dilation=dilation_conf[i],
                      dropout=dropout) for i in range(num_layers)
        ]
        self.tdnn_enc = nn.Sequential(*tdnns)
        self.proj_out = None if output_size is None else nn.Linear(
            dim, output_size)
        sub_sampling = 1
        for s in stride_conf:
            sub_sampling *= s
        self.sub_sampling = sub_sampling

    def forward(self, inp, inp_len):
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None)
        Return:
            out (Tensor): N x To x O
            out_len (Tensor or None)
        """
        out = self.tdnn_enc(inp)
        if inp_len is not None:
            inp_len = inp_len // self.sub_sampling
        if self.proj_out:
            out = self.proj_out(out)
        return out, inp_len


class FSMNEncoder(nn.Module):
    """
    Stack of FsmnLayers, with optional residual connection
    """
    def __init__(self,
                 input_size,
                 output_size,
                 project_size=512,
                 num_layers=4,
                 residual=True,
                 lctx=3,
                 rctx=3,
                 norm="BN",
                 dilation=1,
                 dropout=0):
        super(FSMNEncoder, self).__init__()
        dilations = parse_str_int(dilation, num_layers)
        self.layers = nn.ModuleList([
            FSMNLayer(input_size if i == 0 else output_size,
                      output_size,
                      project_size,
                      lctx=lctx,
                      rctx=rctx,
                      norm="" if i == num_layers - 1 else norm,
                      dilation=dilations[i],
                      dropout=dropout) for i in range(num_layers)
        ])
        self.res = residual

    def forward(self, inp, inp_len):
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
            if self.res:
                inp, memory = fsmn(inp, memory=memory)
            else:
                inp, _ = fsmn(inp, memory=memory)
        inp = inp.transpose(1, 2)
        return inp, inp_len


class TimeDelayRNNEncoder(nn.Module):
    """
    TDNN + RNN encoder (Using TDNN for subsampling and RNN for sequence modeling )
    """
    def __init__(self,
                 input_size,
                 output_size,
                 tdnn_dim=512,
                 tdnn_norm="BN",
                 tdnn_layers=2,
                 tdnn_stride="2,2",
                 tdnn_dilation="1,1",
                 tdnn_dropout=0,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_bidir=True,
                 rnn_dropout=0.2,
                 rnn_project=None,
                 rnn_layernorm=False,
                 rnn_hidden=512):
        super(TimeDelayRNNEncoder, self).__init__()
        self.tdnn_enc = TDNNEncoder(input_size,
                                    None,
                                    dim=tdnn_dim,
                                    norm=tdnn_norm,
                                    num_layers=tdnn_layers,
                                    stride=tdnn_stride,
                                    dilation=tdnn_dilation,
                                    dropout=tdnn_dropout)
        self.rnns_enc = CustomRNNEncoder(tdnn_dim,
                                         output_size,
                                         rnn=rnn,
                                         layernorm=rnn_layernorm,
                                         rnn_layers=rnn_layers,
                                         rnn_bidir=rnn_bidir,
                                         rnn_dropout=rnn_dropout,
                                         rnn_project=rnn_project,
                                         rnn_hidden=rnn_hidden)

    def forward(self, inp, inp_len):
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x To x O
            out_len (Tensor or None): N
        """
        out_pad, out_len = self.tdnn_enc(inp, inp_len)
        # print(f"inp: {inp_len}, out: {out_len}", flush=True)
        return self.rnns_enc(out_pad, out_len)


class TimeDelayFSMNEncoder(nn.Module):
    """
    TDNN + FSMN encoder (Using TDNN for subsampling and FSMN for sequence modeling )
    """
    def __init__(self,
                 input_size,
                 output_size,
                 tdnn_dim=512,
                 tdnn_norm="BN",
                 tdnn_layers=2,
                 tdnn_stride="2,2",
                 tdnn_dilation="1,1",
                 tdnn_dropout=0.2,
                 fsmn_layers=4,
                 fsmn_lctx=10,
                 fsmn_rctx=10,
                 fsmn_norm="LN",
                 fsmn_residual=True,
                 fsmn_dilation=1,
                 fsmn_project=512,
                 fsmn_dropout=0.2):
        super(TimeDelayFSMNEncoder, self).__init__()
        self.tdnn_enc = TDNNEncoder(input_size,
                                    None,
                                    dim=tdnn_dim,
                                    norm=tdnn_norm,
                                    num_layers=tdnn_layers,
                                    stride=tdnn_stride,
                                    dilation=tdnn_dilation,
                                    dropout=tdnn_dropout)
        self.fsmn_enc = FSMNEncoder(tdnn_dim,
                                    output_size,
                                    project_size=fsmn_project,
                                    lctx=fsmn_lctx,
                                    rctx=fsmn_rctx,
                                    norm=fsmn_norm,
                                    dilation=fsmn_dilation,
                                    residual=fsmn_residual,
                                    num_layers=fsmn_layers,
                                    dropout=fsmn_dropout)

    def forward(self, inp, inp_len):
        """
        Args:
            inp (Tensor): N x Ti x F
            inp_len (Tensor or None): N
        Return:
            out_pad (Tensor): N x To x O
            out_len (Tensor or None): N
        """
        out_pad, out_len = self.tdnn_enc(inp, inp_len)
        return self.fsmn_enc(out_pad, out_len)
