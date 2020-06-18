#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from ..asr.base.encoder import TorchEncoder


class UnsupervisedEnh(TorchEncoder):
    """
    A recurrent network for unsupervised training
    """
    def __init__(self,
                 input_size=257,
                 num_bins=257,
                 enh_transform=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.0,
                 rnn_bidir=False):
        super(UnsupervisedEnh, self).__init__(input_size,
                                              num_bins,
                                              rnn=rnn,
                                              rnn_layers=rnn_layers,
                                              rnn_hidden=rnn_hidden,
                                              rnn_dropout=rnn_dropout,
                                              rnn_bidir=rnn_bidir)
        self.enh_transform = enh_transform
        if enh_transform is None:
            raise ValueError("enh_transform can not be None")
        
    def infer(self, noisy):
        """
        Args
            noisy: C x S
        Return
            masks (Tensor): T x F
        """
        with th.no_grad():
            if noisy.dim() != 2:
                raise RuntimeError(
                    "UnsupervisedEnh expects 2D tensor (training), " +
                    f"got {noisy.dim()} instead")
            noisy = noisy[None, ...]
            masks, _ = self.forward(noisy)
            return masks[0]

    def forward(self, noisy):
        """
        Args
            noisy: N x C x S
        Return
            cspec (ComplexTensor): N x C x F x T
            masks (Tensor): N x T x F
        """
        if noisy.dim() != 3:
            raise RuntimeError(
                "UnsupervisedEnh expects 3D tensor (training), " +
                f"got {noisy.dim()} instead")
        # feats: N x T x F
        # cspec: N x C x F x T
        feats, cspec, _ = self.enh_transform(noisy, None, norm_obs=True)
        feats, _ = self.rnns(feats)
        # N x T x F
        masks = th.sigmoid(self.proj(feats))
        return cspec, masks
