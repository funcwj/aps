#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union
from aps.sse.base import SSEBase, MaskNonLinear, tf_masking
from aps.sse.rnn import RNNWrapper
from aps.libs import ApsRegisters


@ApsRegisters.sse.register("sse@chimera++")
class Chimera(SSEBase):
    """
    RNN based Chimera++ Network for Speech Separation
    """

    def __init__(self,
                 input_size: int = 257,
                 input_proj: int = -1,
                 num_bins: int = 257,
                 num_spks: int = 2,
                 enh_transform: Optional[nn.Module] = None,
                 rnn: str = "lstm",
                 num_layers: int = 3,
                 hidden: int = 512,
                 hidden_proj: int = -1,
                 dropout: float = 0.2,
                 dpcl_embed_size: int = 20,
                 bidirectional: bool = False,
                 training_mode: str = "freq",
                 mask_non_linear: str = "sigmoid"):
        super(Chimera, self).__init__(enh_transform,
                                      training_mode=training_mode)
        assert enh_transform is not None
        if num_spks == 1 and mask_non_linear == "softmax":
            raise ValueError(
                "mask_non_linear can not be softmax when num_spks == 1")
        self.encoder = RNNWrapper(input_size,
                                  -1,
                                  input_proj=input_proj,
                                  rnn=rnn,
                                  num_layers=num_layers,
                                  hidden=hidden,
                                  hidden_proj=hidden_proj,
                                  dropout=dropout,
                                  bidirectional=bidirectional,
                                  non_linear="none")
        self.mask_proj = nn.Linear(self.encoder.out_features,
                                   num_spks * num_bins)
        self.dpcl_proj = nn.Linear(self.encoder.out_features,
                                   dpcl_embed_size * num_bins)
        self.dpcl_non_linear = nn.Sigmoid()
        self.mask_non_linear = MaskNonLinear(mask_non_linear, enable="positive")
        self.num_spks = num_spks
        self.embed_size = dpcl_embed_size
        # cached rnn output
        self.cached_rnn_out = None

    def dpcl_embed(self) -> th.Tensor:
        """
        Produce DPCL embeddings (N x FT x D)
        """
        # N x T x D*F
        embed = self.dpcl_proj(self.cached_rnn_out)
        N, T, _ = embed.shape
        # N x T x F x D
        embed = embed.view(N, T, -1, self.embed_size)
        # N x T x F x D
        embed = embed / th.norm(embed, 2, -1, keepdim=True)
        # N x F x T x D
        embed = th.transpose(embed, 1, 2).contiguous()
        # N x FT x D
        return self.dpcl_non_linear(embed.view(N, -1, self.embed_size))

    def _tf_mask(self, feats: th.Tensor, num_spks: int) -> th.Tensor:
        """
        TF mask estimation from given features
        """
        # N x T x D
        self.cached_rnn_out = self.encoder(feats, None)[0]
        # N x T x S*F
        masks = self.mask_proj(self.cached_rnn_out)
        # N x S*F x T
        masks = masks.transpose(1, 2)
        # [N x F x T, ]
        masks = th.chunk(masks, self.num_spks, -2)
        # S x N x F x T
        return self.mask_non_linear(th.stack(masks))

    def _infer(self, mix: th.Tensor,
               mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Running in time or frequency mode and return time signals or frequency TF masks
        """
        # N x F x T x 2
        stft, _ = self.enh_transform.encode(mix, None)
        # feats: N x T x F
        feats = self.enh_transform(stft)
        # S x N x F x T
        masks, embed = self._tf_mask(feats, self.num_spks)
        masks = th.chunk(masks, self.num_spks, 0)
        # [N x F x T, ...]
        masks = [m[0] for m in masks]
        if mode == "freq":
            packed = masks
        else:
            bss_stft = [tf_masking(stft, m) for m in masks]
            packed = self.enh_transform.decode(bss_stft)
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

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x (C) x S
        Return:
            [Tensor, ...]: enhanced signals (N x S) or TF masks (N x F x T)
        """
        self.check_args(mix, training=True, valid_dim=[2, 3])
        return self._infer(mix, self.training_mode)
