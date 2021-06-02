#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union, Dict
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
                 arch: str = "xfmr",
                 arch_kwargs: Dict = {},
                 pose_kwargs: Dict = {},
                 proj_kwargs: Dict = {},
                 num_layers: int = 6,
                 non_linear: str = "sigmoid",
                 mask_dropout: float = 0.1,
                 training_mode: str = "freq") -> None:
        super(FreqRelXfmr, self).__init__(enh_transform,
                                          training_mode=training_mode)
        assert enh_transform is not None
        att_dim = arch_kwargs["att_dim"]
        self.rel_xfmr = TransformerEncoder(arch,
                                           input_size,
                                           num_layers=num_layers,
                                           proj="linear",
                                           proj_kwargs=proj_kwargs,
                                           pose="rel",
                                           pose_kwargs=pose_kwargs,
                                           arch_kwargs=arch_kwargs)
        self.proj = nn.Sequential(nn.Linear(input_size, att_dim),
                                  nn.LayerNorm(att_dim),
                                  nn.Dropout(mask_dropout))
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
