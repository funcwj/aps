#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union
from aps.asr.xfmr.encoder import TransformerEncoder
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("sse@freq_xfmr_rel")
class FreqRelXfmr(SseBase):
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
                 radius: int = 256,
                 feedforward_dim: int = 2048,
                 att_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 proj_dropout: float = 0.1,
                 post_norm: bool = True,
                 num_layers: int = 6,
                 non_linear: str = "sigmoid",
                 training_mode: str = "freq") -> None:
        super(FreqRelXfmr, self).__init__(enh_transform,
                                          training_mode=training_mode)
        assert enh_transform is not None
        self.rel_xfmr = TransformerEncoder("xfmr_rel",
                                           input_size,
                                           proj_layer="linear",
                                           att_dim=att_dim,
                                           radius=radius,
                                           nhead=nhead,
                                           feedforward_dim=feedforward_dim,
                                           scale_embed=False,
                                           ffn_dropout=ffn_dropout,
                                           att_dropout=att_dropout,
                                           post_norm=post_norm,
                                           num_layers=num_layers)
        self.proj = nn.Sequential(nn.Linear(input_size, att_dim),
                                  nn.LayerNorm(att_dim),
                                  nn.Dropout(proj_dropout))
        self.mask = nn.Linear(att_dim, num_bins * num_spks)
        self.non_linear = MaskNonLinear(non_linear,
                                        enable="positive_wo_softmax")
        self.num_spks = num_spks

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False, valid_dim=[1])
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
        out, _ = self.rel_xfmr(feats, None)
        # N x T x F
        mask = self.non_linear(self.mask(out))
        # N x F x T
        mask = mask.transpose(1, 2)
        if self.num_spks > 1:
            mask = th.chunk(mask, self.num_spks, 1)
        if mode == "freq":
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
        self.check_args(s, training=True, valid_dim=[2])
        return self._forward(s, mode=self.training_mode)
