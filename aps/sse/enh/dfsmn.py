#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from aps.asr.base.encoder import FSMNEncoder
from aps.transform.asr import TFTransposeTransform
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters
from typing import Optional, Union, List


@ApsRegisters.sse.register("sse@dfsmn")
class DFSMN(SseBase):
    """
    Uses Deep FSMN for speech enhancement/separation
    """

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 dim: int = 1024,
                 num_bins: int = 257,
                 num_spks: int = 1,
                 num_layers: int = 4,
                 project: int = 512,
                 dropout: float = 0.0,
                 residual: bool = True,
                 lcontext: int = 3,
                 rcontext: int = 3,
                 norm: str = "BN",
                 dilation: Union[List[int], int] = 1,
                 non_linear: str = "relu",
                 training_mode: str = "freq"):
        super(DFSMN, self).__init__(enh_transform, training_mode=training_mode)
        assert enh_transform is not None
        self.dfsmn = FSMNEncoder(num_bins,
                                 num_bins * num_spks,
                                 dim=dim,
                                 project=project,
                                 dropout=dropout,
                                 num_layers=num_layers,
                                 residual=residual,
                                 lcontext=lcontext,
                                 rcontext=rcontext,
                                 norm=norm,
                                 dilation=dilation)
        self.masks = nn.Sequential(MaskNonLinear(non_linear, enable="common"),
                                   TFTransposeTransform())
        self.num_spks = num_spks

    def _tf_mask(self, feats: th.Tensor, num_spks: int) -> List[th.Tensor]:
        """
        TF mask estimation from given features
        """
        proj, _ = self.dfsmn(feats, None)
        # N x S*F x T
        masks = self.masks(proj)
        # [N x F x T, ...]
        return th.chunk(masks, self.num_spks, 1)

    def _infer(self, mix: th.Tensor,
               mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Return time signals or frequency TF masks
        """
        # mix_stft: N x F x T
        feats, mix_stft, _ = self.enh_transform(mix, None)
        # [N x F x T, ...]
        masks = self._tf_mask(feats, self.num_spks)
        # post processing
        if mode == "time":
            decoder = self.enh_transform.inverse_stft
            bss_stft = [mix_stft * m for m in masks]
            packed = [
                decoder(s.as_real(), return_polar=False) for s in bss_stft
            ]
        else:
            packed = masks
        return packed[0] if self.num_spks == 1 else packed

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S, mixture signals
        Return:
            [Tensor, ...]: enhanced signals or TF masks
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            ret = self._infer(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    @th.jit.ignore
    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S, mixture signals
        Return:
            [Tensor, ...]: enhanced signals or TF masks
        """
        self.check_args(mix, training=True, valid_dim=[2])
        return self._infer(mix, self.training_mode)

    @th.jit.export
    def mask_predict(self, feats: th.Tensor) -> th.Tensor:
        """
        Args:
            feats (Tensor): noisy feature, N x T x F
        Return:
            masks (Tensor): masks of each speaker, N x T x F
        """
        masks = self._tf_mask(feats, self.num_spks)
        # S x N x F x T
        masks = th.stack(masks)
        return masks[0] if self.num_spks == 1 else masks
