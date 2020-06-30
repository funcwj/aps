#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BaseEncoder(nn.Module):
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
        super(BaseEncoder, self).__init__()
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
        self.mask = support_non_linear[non_linear]

    def check_args(self, mix, training=True):
        """
        Check args training | inference
        """
        if not training and mix.dim() not in [1, 2]:
            raise RuntimeError(
                f"{self.__class__.__name__} expects 1/2D tensor (inference), "
                + f"got {mix.dim()} instead")
        if training and mix.dim() not in [2, 3]:
            raise RuntimeError(
                f"{self.__class__.__name__} expects 2/3D tensor (training), " +
                f"got {mix.dim()} instead")

    def forward(self, inp_pad, inp_len, max_len=None):
        """
        Args:
            inp_pad (Tensor): (N) x Ti x F
            inp_len (Tensor): (N) x Ti
        Return:
            y_pad (Tensor): (N) x Ti x F
            y_len (Tensor): (N) x Ti
        """
        self.rnns.flatten_parameters()
        if inp_len is not None:
            inp_pad = pack_padded_sequence(inp_pad, inp_len, batch_first=True)
        # extend dim when inference
        else:
            if inp_pad.dim() not in [2, 3]:
                raise RuntimeError("BaseEncoder expects 2/3D Tensor, " +
                                   f"got {inp_pad.dim():d}")
            if inp_pad.dim() != 3:
                inp_pad = th.unsqueeze(inp_pad, 0)
        if self.proj:
            inp_pad = tf.relu(self.proj(inp_pad))
        y, _ = self.rnns(inp_pad)
        # using unpacked sequence
        # y: NxTxD
        if inp_len is not None:
            y, _ = pad_packed_sequence(y,
                                       batch_first=True,
                                       total_length=max_len)
        y = self.outp(y)
        # pass through non-linear
        if self.mask:
            y = self.mask(y)
        return y, inp_len


class FreqDomainToyRNN(BaseEncoder):
    """
    Toy RNN structure for separation & enhancement (frequency domain)
    """
    def __init__(self,
                 input_size=257,
                 input_project=None,
                 num_bins=257,
                 num_spks=2,
                 enh_transform=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.2,
                 rnn_bidir=False,
                 output_nonlinear="sigmoid"):
        super(FreqDomainToyRNN, self).__init__(input_size,
                                               num_bins * num_spks,
                                               input_project=input_project,
                                               rnn=rnn,
                                               rnn_layers=rnn_layers,
                                               rnn_hidden=rnn_hidden,
                                               rnn_dropout=rnn_dropout,
                                               rnn_bidir=rnn_bidir,
                                               non_linear=output_nonlinear)
        if enh_transform is None:
            raise ValueError("enh_transform can not be None")
        self.enh_transform = enh_transform
        self.num_spks = num_spks

    def infer(self, mix):
        """
        Args:
            mix (Tensor): (C) x S
        Return:
            sep [Tensor, ...]: S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, ...]
            # feats: N x T x F
            feats, stft, _ = self.enh_transform(mix, None)
            # use ch0 as reference if multi-channel
            if stft.dim() == 4:
                stft = stft[:, 0]
            # N x T x 2F
            masks, _ = super().forward(feats, None)
            # N x 2F x T
            masks = masks.transpose(1, 2)
            # [N x F x T, ...]
            masks = th.chunk(masks, self.num_spks, 1)
            # complex tensor
            spk_stft = [stft * m for m in masks]
            spk = [
                self.enh_transform.inverse_stft((s.real, s.imag),
                                                input="complex")
                for s in spk_stft
            ]
            if self.num_spks == 1:
                return spk[0]
            else:
                return [s[0] for s in spk]

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            masks [Tensor, ...]: N x F x T
        """
        self.check_args(mix, training=True)
        # feats: N x T x F
        feats, _, _ = self.enh_transform(mix, None)
        # N x T x 2F
        masks, _ = super().forward(feats, None)
        # N x *F x T
        masks = masks.transpose(1, 2)
        if self.num_spks == 1:
            return masks
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        return masks


class TimeDomainToyRNN(BaseEncoder):
    """
    Toy RNN structure for separation & enhancement (time domain)
    """
    def __init__(self,
                 input_size=257,
                 input_project=None,
                 num_bins=257,
                 num_spks=2,
                 enh_transform=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.2,
                 rnn_bidir=False,
                 output_nonlinear="sigmoid"):
        super(TimeDomainToyRNN, self).__init__(input_size,
                                               num_bins * num_spks,
                                               input_project=input_project,
                                               rnn=rnn,
                                               rnn_layers=rnn_layers,
                                               rnn_hidden=rnn_hidden,
                                               rnn_dropout=rnn_dropout,
                                               rnn_bidir=rnn_bidir,
                                               non_linear=output_nonlinear)
        if enh_transform is None:
            raise ValueError("enh_transform can not be None")
        self.enh_transform = enh_transform
        self.num_spks = num_spks

    def infer(self, mix):
        """
        Args:
            mix (Tensor): (C) x S
        Return:
            sep [Tensor, ...]: S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            # N x (C) x S
            mix = mix[None, ...]
            sep = self.forward(mix)
            if isinstance(sep, th.Tensor):
                return sep[0]
            else:
                return [s[0] for s in sep]

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            sep [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True)
        # feats: N x T x F
        # stft: N x (C) x F x T
        feats, stft, _ = self.enh_transform(mix, None)
        # use ch0 as reference if multi-channel
        if stft.dim() == 4:
            stft = stft[:, 0]
        # N x T x 2F
        masks, _ = super().forward(feats, None)
        # N x 2F x T
        masks = masks.transpose(1, 2)
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        # complex tensor
        spk_stft = [stft * m for m in masks]
        spk = [
            self.enh_transform.inverse_stft((s.real, s.imag), input="complex")
            for s in spk_stft
        ]
        if self.num_spks == 1:
            return spk[0]
        else:
            return spk
