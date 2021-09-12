# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Union, List, Optional, Dict
from aps.rt_sse.base import RealTimeSSEBase
from aps.sse.base import tf_masking, MaskNonLinear
from aps.transform.asr import TFTransposeTransform
from aps.streaming_asr.transformer.encoder import StreamingTransformerEncoder
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("rt_sse@freq_xfmr")
class FreqXfmr(RealTimeSSEBase):
    """
    Transformer for speech enhancement/separation
    """

    __constants__ = ["chunk", "complex_mask"]

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 num_bins: int = 257,
                 num_branchs: int = 1,
                 num_layers: int = 6,
                 chunk: int = 1,
                 lctx: int = 3,
                 arch: str = "xfmr",
                 proj_kwargs: Dict = {},
                 pose: str = "rel",
                 pose_kwargs: Dict = {},
                 arch_kwargs: Dict = {},
                 complex_mask: bool = True,
                 non_linear: str = "relu",
                 training_mode: str = "freq"):
        super(FreqXfmr, self).__init__(enh_transform,
                                       training_mode=training_mode)
        output_dim = num_bins * num_branchs * (2 if complex_mask else 1)
        self.xfmr = StreamingTransformerEncoder(arch,
                                                num_bins,
                                                output_proj=output_dim,
                                                num_layers=num_layers,
                                                chunk=chunk,
                                                lctx=lctx,
                                                proj="linear",
                                                proj_kwargs=proj_kwargs,
                                                pose="rel",
                                                pose_kwargs=pose_kwargs,
                                                arch_kwargs=arch_kwargs)
        if complex_mask:
            self.masks = nn.Sequential(MaskNonLinear("none", enable="all"),
                                       TFTransposeTransform())
        else:
            self.masks = nn.Sequential(
                MaskNonLinear(non_linear, enable="common"),
                TFTransposeTransform())
        self.num_branchs = num_branchs
        self.complex_mask = complex_mask
        self.chunk = chunk

    def _tf_mask(self, feats: th.Tensor) -> List[th.Tensor]:
        """
        TF mask estimation from given features
        """
        proj = self.xfmr(feats, None)[0]
        # N x S*F x T
        masks = self.masks(proj)
        # [N x F x T, ...]
        return th.chunk(masks, self.num_branchs, 1)

    def _infer(self, mix: th.Tensor,
               mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Return time signals or frequency TF masks
        """
        # stft: N x F x T x 2
        stft, _ = self.enh_transform.encode(mix, None)
        # N x T x F
        feats = self.enh_transform(stft)
        # [N x F x T, ...]
        masks = self._tf_mask(feats)
        if self.complex_mask:
            # [N x F x T x 2, ...]
            masks = [th.stack(th.chunk(m, 2, 1), -1) for m in masks]
        # post processing
        if mode == "time":
            bss_stft = [tf_masking(stft, m) for m in masks]
            packed = self.enh_transform.decode(bss_stft)
        else:
            packed = masks
        return packed[0] if self.num_branchs == 1 else packed

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
            return ret[0] if self.num_branchs == 1 else [r[0] for r in ret]

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
    def reset(self):
        self.xfmr.reset()

    @th.jit.export
    def step(self, chunk: th.Tensor) -> th.Tensor:
        """
        Processing one step
        """
        # N x S*F x T
        masks = self.masks(self.xfmr.step(chunk))
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_branchs, 1)
        if self.complex_mask:
            # [N x F x T x 2, ...]
            masks = [th.stack(th.chunk(m, 2, 1), -1) for m in masks]
        # S x N x F x T or S x N x F x T x 2
        masks = th.stack(masks)
        return masks[0] if self.num_branchs == 1 else masks
