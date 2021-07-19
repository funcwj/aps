#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Union, List
from aps.asr.base.encoder import PyTorchRNNEncoder
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("sse@base_rnn")
class ToyRNN(SseBase):
    """
    Toy RNN structure for separation & enhancement
    """

    def __init__(self,
                 input_size: int = 257,
                 input_proj: Optional[int] = None,
                 num_bins: int = 257,
                 num_spks: int = 2,
                 enh_transform: Optional[nn.Module] = None,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 dropout: float = 0.2,
                 bidirectional: bool = False,
                 output_nonlinear: str = "sigmoid",
                 training_mode: str = "freq") -> None:
        super(ToyRNN, self).__init__(enh_transform, training_mode=training_mode)
        assert enh_transform is not None
        if num_spks == 1 and output_nonlinear == "softmax":
            raise ValueError(
                "output_nonlinear can not be softmax when num_spks == 1")
        self.base_rnn = PyTorchRNNEncoder(input_size,
                                          num_bins * num_spks,
                                          input_project=input_proj,
                                          rnn=rnn,
                                          num_layers=num_layers,
                                          hidden=hidden,
                                          dropout=dropout,
                                          bidirectional=bidirectional,
                                          non_linear="none")
        self.non_linear = MaskNonLinear(output_nonlinear, enable="positive")
        self.num_spks = num_spks

    def _tf_mask(self, feats: th.Tensor, num_spks: int) -> th.Tensor:
        """
        TF mask estimation from given features
        """
        # N x T x S*F
        masks, _ = self.base_rnn(feats, None)
        # N x S*F x T
        masks = masks.transpose(1, 2)
        # [N x F x T, ]
        masks = th.chunk(masks, self.num_spks, -2)
        # S x N x F x T
        return self.non_linear(th.stack(masks))

    def _infer(self, mix: th.Tensor,
               mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Running in time or frequency mode and return time signals or frequency TF masks
        """
        # feats: N x T x F
        feats, stft, _ = self.enh_transform(mix, None)
        N, T, _ = feats.shape
        if stft.dim() == 4:
            # N x F x T
            stft = stft[:, 0]
        # S x N x F x T
        masks = self._tf_mask(feats, self.num_spks)
        masks = th.chunk(masks, self.num_spks, 0)
        # [N x F x T, ...]
        masks = [m[0] for m in masks]
        if mode == "freq":
            packed = masks
        else:
            decoder = self.enh_transform.inverse_stft
            bss_stft = [stft * m for m in masks]
            packed = [
                decoder(s.as_real(), return_polar=False) for s in bss_stft
            ]
        return packed[0] if self.num_spks == 1 else packed

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): (C) x S
        Return:
            [Tensor, ...]: enhanced signals or TF masks
        """
        self.check_args(mix, training=False, valid_dim=[1, 2])
        with th.no_grad():
            mix = mix[None, ...]
            spk = self._infer(mix, mode)
            return spk[0] if self.num_spks == 1 else [s[0] for s in spk]

    @th.jit.ignore
    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            [Tensor, ...]: enhanced signals (N x S) or TF masks (N x F x T)
        """
        self.check_args(mix, training=True, valid_dim=[2, 3])
        return self._infer(mix, self.training_mode)

    @th.jit.export
    def mask_predict(self, feats: th.Tensor) -> th.Tensor:
        """
        Args:
            feats (Tensor): noisy feature, N x T x F
        Return:
            masks (Tensor): masks of each speaker, N x T x F
        """
        return self._tf_mask(feats, self.num_spks)
