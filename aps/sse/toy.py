#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, NoReturn, Union, List
from aps.asr.base.encoder import TorchRNNEncoder


class ToyRNN(TorchRNNEncoder):
    """
    Toy RNN structure for separation & enhancement
    """

    def __init__(self,
                 input_size: int = 257,
                 input_project: Optional[int] = None,
                 num_bins: int = 257,
                 num_spks: int = 2,
                 enh_transform: Optional[nn.Module] = None,
                 rnn: str = "lstm",
                 rnn_layers: int = 3,
                 rnn_hidden: int = 512,
                 rnn_dropout: float = 0.2,
                 rnn_bidir: bool = False,
                 output_nonlinear: str = "sigmoid",
                 training_mode: str = "freq") -> None:
        super(ToyRNN, self).__init__(input_size,
                                     num_bins * num_spks,
                                     input_project=input_project,
                                     rnn=rnn,
                                     rnn_layers=rnn_layers,
                                     rnn_hidden=rnn_hidden,
                                     rnn_dropout=rnn_dropout,
                                     rnn_bidir=rnn_bidir,
                                     non_linear=output_nonlinear
                                     if output_nonlinear != "softmax" else "")
        if enh_transform is None:
            raise ValueError("enh_transform can not be None")
        if num_spks == 1 and output_nonlinear == "softmax":
            raise ValueError(
                "output_nonlinear can not be softmax when num_spks == 1")
        self.enh_transform = enh_transform
        self.num_spks = num_spks
        self.output_nonlinear = output_nonlinear
        self.mode = training_mode

    def check_args(self, mix: th.Tensor, training: bool = True) -> NoReturn:
        """
        Check args training | inference
        """
        if not training and mix.dim() not in [1, 2]:
            raise RuntimeError("{ToyRNN expects 1/2D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() not in [2, 3]:
            raise RuntimeError("ToyRNN expects 2/3D tensor (training), " +
                               f"got {mix.dim()} instead")

    def _forward(self, mix: th.Tensor,
                 mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
            time mode: return time domain signal
            freq mode: return TF-mask
        """
        # feats: N x T x F
        feats, stft, _ = self.enh_transform(mix, None)
        N, T, _ = feats.shape
        if stft.dim() == 4:
            stft = stft[:, 0]
        # N x T x 2F
        masks, _ = super().forward(feats, None)
        # N x *F x T
        masks = masks.transpose(1, 2)
        # [N x F x T, ...]
        if self.num_spks > 1:
            if self.output_nonlinear == "softmax":
                # N x S x F x T
                masks = masks.view(N, self.num_spks, -1, T)
                masks = tf.softmax(masks, dim=1)
                masks = masks.view(N, -1, T)
            masks = th.chunk(masks, self.num_spks, 1)
        if mode == "freq":
            return masks
        else:
            if self.num_spks == 1:
                masks = [masks]
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
                return spk

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): (C) x S
        Return:
            sep [Tensor, ...]: S or
            masks [Tensor, ...]: F x T
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, ...]
            spk = self._forward(mix, mode)
            return spk[0] if self.num_spks == 1 else [s[0] for s in spk]

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            masks [Tensor, ...]: N x F x T or
            spks [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True)
        return self._forward(mix, self.mode)
