#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..asr.base.encoder import TorchEncoder


class FreqDomainToyRNN(TorchEncoder):
    """
    Toy RNN structure for separation & enhancement (frequency domain)
    """
    def __init__(self,
                 input_size=257,
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
                                               rnn=rnn,
                                               rnn_layers=rnn_layers,
                                               rnn_hidden=rnn_hidden,
                                               rnn_dropout=rnn_dropout,
                                               rnn_bidir=rnn_bidir,
                                               output_act=output_nonlinear)
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
        with th.no_grad():
            if mix.dim() not in [1, 2]:
                raise RuntimeError("ToyRNN expects 1/2D tensor (inference), " +
                                   f"got {mix.dim()} instead")
            mix = mix[None, ...]
            # feats: N x T x F
            feats, stft, _ = self.enh_transform(mix, None)
            # use ch0 as reference if multi-channel
            if stft.dim() == 4:
                stft = stft[:, 0]
            rnn_out, _ = self.rnns(feats)
            # N x T x 2F
            masks = self.oact(self.proj(rnn_out))
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
        if mix.dim() not in [2, 3]:
            raise RuntimeError("ToyRNN expects 2/3D tensor (training), " +
                               f"got {mix.dim()} instead")
        # feats: N x T x F
        feats, _, _ = self.enh_transform(mix, None)
        rnn_out, _ = self.rnns(feats)
        # N x T x *F
        masks = self.oact(self.proj(rnn_out))
        # N x *F x T
        masks = masks.transpose(1, 2)
        if self.num_spks == 1:
            return masks
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        return masks


class TimeDomainToyRNN(TorchEncoder):
    """
    Toy RNN structure for separation & enhancement (time domain)
    """
    def __init__(self,
                 input_size=257,
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
                                               rnn=rnn,
                                               rnn_layers=rnn_layers,
                                               rnn_hidden=rnn_hidden,
                                               rnn_dropout=rnn_dropout,
                                               rnn_bidir=rnn_bidir,
                                               output_act=output_nonlinear)
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
        with th.no_grad():
            if mix.dim() not in [1, 2]:
                raise RuntimeError("ToyRNN expects 1/2D tensor (inference), " +
                                   f"got {mix.dim()} instead")
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
        if mix.dim() not in [2, 3]:
            raise RuntimeError("ToyRNN expects 2/3D tensor (training), " +
                               f"got {mix.dim()} instead")
        # feats: N x T x F
        # stft: N x (C) x F x T
        feats, stft, _ = self.enh_transform(mix, None)
        # use ch0 as reference if multi-channel
        if stft.dim() == 4:
            stft = stft[:, 0]
        rnn_out, _ = self.rnns(feats)
        # N x T x 2F
        masks = self.oact(self.proj(rnn_out))
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
