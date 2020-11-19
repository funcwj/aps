#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union, NoReturn
from aps.asr.transformer.encoder import RelTransformerEncoder
from aps.sse.utils import MaskNonLinear
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("freq_rel_transformer")
class FreqRelTransformer(RelTransformerEncoder):
    """
    Frequency domain Transformer model
    """

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 input_size: int = 257,
                 num_spks: int = 2,
                 num_bins: int = 257,
                 att_dim: int = 512,
                 nhead: int = 8,
                 k_dim: int = 256,
                 feedforward_dim: int = 2048,
                 att_dropout: float = 0.1,
                 proj_dropout: float = 0.1,
                 post_norm: bool = True,
                 add_value_rel: bool = False,
                 num_layers: int = 6,
                 non_linear: str = "sigmoid",
                 training_mode: str = "freq") -> None:
        super(FreqRelTransformer,
              self).__init__(input_size,
                             input_embed="linear",
                             att_dim=att_dim,
                             k_dim=k_dim,
                             nhead=nhead,
                             feedforward_dim=feedforward_dim,
                             scale_embed=False,
                             pos_dropout=0,
                             att_dropout=att_dropout,
                             post_norm=post_norm,
                             add_value_rel=add_value_rel,
                             num_layers=num_layers)
        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        self.enh_transform = enh_transform
        self.mode = training_mode
        self.proj = nn.Sequential(nn.Linear(input_size, att_dim),
                                  nn.LayerNorm(att_dim),
                                  nn.Dropout(proj_dropout))
        self.mask = nn.Linear(att_dim, num_bins * num_spks)
        self.non_linear = MaskNonLinear(non_linear,
                                        enable="positive_wo_softmax")
        self.num_spks = num_spks

    def check_args(self, mix: th.Tensor, training: bool = True) -> NoReturn:
        if not training and mix.dim() != 1:
            raise RuntimeError(
                "FreqRelTransformer expects 1D tensor (inference), " +
                f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError(
                "FreqRelTransformer expects 2D tensor (training), " +
                f"got {mix.dim()} instead")

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            sep = self._forward(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def _forward(self,
                 mix: th.Tensor,
                 mode: str = "freq") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
        """
        feats, stft, _ = self.enh_transform(mix, None)
        # stft: N x F x T
        out, _ = super().forward(feats, None)
        # T x N x F => N x T x F
        out = out.transpose(0, 1)
        # N x T x F
        mask = self.non_linear(self.mask(out))
        # N x F x T
        mask = mask.transpose(1, 2)
        if self.num_spks > 1:
            mask = th.chunk(mask, self.num_spks, 1)
        if self.mode == "freq":
            return mask
        else:
            decoder = self.enh_transform.inverse_stft
            if self.num_spks == 1:
                mask = [mask]
            # complex tensor
            spk_stft = [stft * m for m in mask]
            spk = [decoder((s.real, s.imag), input="complex") for s in spk_stft]
            if self.num_spks == 1:
                return spk[0]
            else:
                return spk

    def forward(self, s: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True)
        return self._forward(s, mode=self.mode)
