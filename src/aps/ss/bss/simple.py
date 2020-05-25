#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ...asr.base.encoder import TorchEncoder


class Simple(TorchEncoder):
    """
    A recurrent network for unsupervised training
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
                 output_nonlinear="sigmoid",
                 time_domain=True):
        if enh_transform is None:
            raise ValueError("Simple: enh_transform can not be None")
        super(Simple, self).__init__(input_size,
                                     num_bins * num_spks,
                                     rnn=rnn,
                                     rnn_layers=rnn_layers,
                                     rnn_hidden=rnn_hidden,
                                     rnn_dropout=rnn_dropout,
                                     rnn_bidir=rnn_bidir,
                                     output_act=output_nonlinear)
        self.enh_transform = enh_transform
        self.num_spks = num_spks
        if time_domain:
            self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        else:
            self.inverse_stft = None

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        if s.dim() not in [1, 2]:
            raise RuntimeError(f"Expect 1/2D tensor, got {s.dim()} instead")
        if s.dim() == 1:
            s = s[None, ...]
        # feats: N x T x F
        # stft: N x F x T
        feats, stft, _ = self.enh_transform(s, None)
        rnn_out, _ = self.rnns(feats)
        # N x T x 2F
        masks = self.oact(self.proj(rnn_out))
        # N x 2F x T
        masks = masks.transpose(1, 2)
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        if self.inverse_stft:
            # complex tensor
            spk_stft = [stft * m for m in masks]
            return [
                self.inverse_stft((s.real, s.imag), input="complex")
                for s in spk_stft
            ]
        return masks
