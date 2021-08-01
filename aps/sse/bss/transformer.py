#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union, Dict
from aps.asr.transformer.encoder import TransformerEncoder
from aps.sse.base import SseBase, MaskNonLinear
from aps.transform.asr import TFTransposeTransform
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("sse@freq_xfmr")
class FreqXfmr(SseBase):
    """
    Frequency domain Transformer based model
    """

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 input_size: int = 257,
                 num_spks: int = 2,
                 num_bins: int = 257,
                 rctx: int = -1,
                 lctx: int = -1,
                 arch: str = "xfmr",
                 pose: str = "rel",
                 arch_kwargs: Dict = {},
                 pose_kwargs: Dict = {},
                 proj_kwargs: Dict = {},
                 num_layers: int = 6,
                 non_linear: str = "sigmoid",
                 training_mode: str = "freq") -> None:
        super(FreqXfmr, self).__init__(enh_transform,
                                       training_mode=training_mode)
        assert enh_transform is not None
        att_dim = arch_kwargs["att_dim"]
        self.xfmr = TransformerEncoder(arch,
                                       input_size,
                                       num_layers=num_layers,
                                       chunk_size=1,
                                       lctx=lctx,
                                       rctx=rctx,
                                       proj="linear",
                                       proj_kwargs=proj_kwargs,
                                       pose=pose,
                                       pose_kwargs=pose_kwargs,
                                       arch_kwargs=arch_kwargs)
        self.mask = nn.Sequential(nn.Linear(att_dim, num_bins * num_spks),
                                  MaskNonLinear(non_linear, enable="common"),
                                  TFTransposeTransform())
        self.num_spks = num_spks

    def _tf_mask(self, feats: th.Tensor, num_spks: int) -> List[th.Tensor]:
        """
        TF mask estimation from given features
        """
        # stft: N x F x T
        out, _ = self.xfmr(feats, None)
        # N x T x F => N x S*F x T
        mask = self.mask(out)
        # [N x F x T, ...]
        return th.chunk(mask, self.num_spks, 1)

    def _infer(self,
               mix: th.Tensor,
               mode: str = "freq") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Running in time or frequency mode and return time signals or frequency TF masks
        """
        feats, stft, _ = self.enh_transform(mix, None)
        masks = self._tf_mask(feats, self.num_spks)
        # post processing
        if mode == "time":
            decoder = self.enh_transform.inverse_stft
            bss_stft = [stft * m for m in masks]
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
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            sep = self._infer(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    @th.jit.ignore
    def forward(self, s: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True, valid_dim=[2])
        return self._infer(s, mode=self.training_mode)

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
