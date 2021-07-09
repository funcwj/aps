# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.transform.asr import TFTransposeTransform
from aps.asr.xfmr.encoder import TransformerEncoder
from aps.sse.base import SseBase, MaskNonLinear
from aps.sse.bss.tcn import normalize_layer
from aps.libs import ApsRegisters

from typing import Dict, List, Union, Optional


class Transformer(nn.Module):
    """
    For wrapping of the inter/intra Transformer
    """

    def __init__(self,
                 arch: str = "xfmr",
                 num_layers: int = 2,
                 arch_kwargs: Dict = {}):
        super(Transformer, self).__init__()
        self.transformer = TransformerEncoder(arch,
                                              -1,
                                              num_layers=num_layers,
                                              proj="none",
                                              pose="abs",
                                              arch_kwargs=arch_kwargs)

    def forward(self, chunk: th.Tensor) -> th.Tensor:
        """
        Sequence modeling along axis K
        Args:
            chunk (Tensor): N x K x L x C
        Return:
            chunk (Tensor): N x L x K x C
        """
        # N x K x L x C
        N, K, L, C = chunk.shape
        # N x L x K x C
        chunk = chunk.transpose(1, 2)
        # NL x K x C
        chunk = chunk.contiguous()
        chunk = chunk.view(-1, K, C)
        # K x NL x C
        chunk, _ = self.transformer(chunk, None)
        # NL x K x C
        chunk = chunk.transpose(0, 1)
        # N x L x K x C
        return chunk.view(N, L, K, C)


class SepFormer(nn.Module):
    """
    Main network of SepFormer proposed in Attention is All You Need in Speech Separation
    """

    def __init__(self,
                 arch: str,
                 num_bins: int = 256,
                 num_spks: int = 2,
                 num_blocks: int = 2,
                 num_layers: int = 2,
                 chunk_size: int = 320,
                 arch_kwargs: Dict = {}):
        super(SepFormer, self).__init__()
        self.chunk_size = chunk_size
        xfmr_kwargs = {
            "arch": arch,
            "num_layers": num_layers,
            "arch_kwargs": arch_kwargs
        }
        separator = []
        separator += [nn.Linear(num_bins, arch_kwargs["att_dim"])]
        # [intra, inter, intra, inter, ...]
        separator += [Transformer(**xfmr_kwargs) for _ in range(num_blocks * 2)]
        separator += [nn.PReLU(), nn.Linear(arch_kwargs["att_dim"], num_bins)]
        self.separator = nn.Sequential(*separator)
        self.mask = nn.Conv1d(num_bins, num_bins * num_spks, 1)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T
        Return:
            masks (Tensor): N x S*C x T
        """
        batch_size, num_bins, num_frames = inp.shape
        # N x C x T x 1 => N x CK x L
        chunks = tf.unfold(inp[..., None], (self.chunk_size, 1),
                           stride=self.chunk_size // 2)
        # N x C x K x L
        chunks = chunks.view(batch_size, num_bins, self.chunk_size, -1)
        # N x L x K x C
        chunks = chunks.transpose(1, -1)
        # N x K x L x C
        chunks = self.separator(chunks)
        # N x C x K x L
        chunks = chunks.transpose(1, -1)
        # N x CK x L
        chunks = chunks.contiguous()
        chunks = chunks.view(batch_size, -1, chunks.shape[-1])
        # N x C x T x 1
        out = tf.fold(chunks, (num_frames, 1), (self.chunk_size, 1),
                      stride=self.chunk_size // 2)
        # N x C*S x T
        return self.mask(out[..., 0])


@ApsRegisters.sse.register("sse@time_sepformer")
class TimeSeqFormer(SseBase):
    """
    SeqFormer network in time domain
    """

    def __init__(self,
                 arch: str = "xfmr",
                 stride: int = 8,
                 kernel: int = 16,
                 num_bins: int = 256,
                 num_spks: int = 2,
                 non_linear: str = "relu",
                 num_blocks: int = 2,
                 num_layers: int = 2,
                 chunk_size: int = 320,
                 arch_kwargs: Dict = {}):
        super(TimeSeqFormer, self).__init__(None, training_mode="time")
        self.encoder = nn.Conv1d(1,
                                 num_bins,
                                 kernel_size=kernel,
                                 stride=stride,
                                 padding=0)
        self.norm = normalize_layer("cLN", num_bins)
        self.separator = SepFormer(arch,
                                   num_bins=num_bins,
                                   num_spks=num_spks,
                                   num_blocks=num_blocks,
                                   num_layers=num_layers,
                                   chunk_size=chunk_size,
                                   arch_kwargs=arch_kwargs)
        self.mask = MaskNonLinear(non_linear, enable="positive_wo_softmax")
        self.decoder = nn.ConvTranspose1d(num_bins,
                                          1,
                                          kernel_size=kernel,
                                          stride=stride,
                                          bias=True)
        self.num_spks = num_spks

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            sep ([Tensor, ...]): S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, ...]
            sep = self.forward(mix)
            return sep[0] if self.num_spks == 1 else [s[0] for s in sep]

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        # N x 1 x S => N x C x T
        w = self.norm(th.relu(self.encoder(mix[:, None])))
        # N x C*S x T
        m = self.mask(self.separator(w))
        # [N x C x T, ...]
        m = th.chunk(m, self.num_spks, 1)
        # S x N x C x T
        m = th.stack(m, dim=0)
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        bss = [self.decoder(x)[:, 0] for x in s]
        return bss[0] if self.num_spks == 1 else bss


@ApsRegisters.sse.register("sse@freq_sepformer")
class FreqSeqFormer(SseBase):
    """
    SeqFormer network in frequency domain
    """

    def __init__(self,
                 arch: str = "xfmr",
                 enh_transform: Optional[nn.Module] = None,
                 num_bins: int = 257,
                 num_spks: int = 2,
                 non_linear: str = "relu",
                 num_blocks: int = 2,
                 num_layers: int = 2,
                 chunk_size: int = 64,
                 arch_kwargs: Dict = {},
                 training_mode: str = "freq"):
        super(FreqSeqFormer, self).__init__(enh_transform,
                                            training_mode=training_mode)
        assert enh_transform is not None
        self.swap = TFTransposeTransform()
        self.separator = SepFormer(arch,
                                   num_bins=num_bins,
                                   num_spks=num_spks,
                                   num_blocks=num_blocks,
                                   num_layers=num_layers,
                                   chunk_size=chunk_size,
                                   arch_kwargs=arch_kwargs)
        self.mask = MaskNonLinear(non_linear, enable="common")
        self.num_spks = num_spks

    def _forward(self, mix: th.Tensor,
                 mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
        """
        # mix_stft: N x F x T
        feats, mix_stft, _ = self.enh_transform(mix, None)
        # N x S*F x T
        masks = self.mask(self.separator(self.swap(feats)))
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        if mode == "time":
            decoder = self.enh_transform.inverse_stft
            bss_stft = [mix_stft * m for m in masks]
            bss = [decoder((s.real, s.imag), input="complex") for s in bss_stft]
        else:
            bss = masks
        return bss[0] if self.num_spks == 1 else bss

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=False, valid_dim=[1, 2])
        with th.no_grad():
            mix = mix[None, :]
            ret = self._forward(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args
            mix (Tensor): N x (C) x S
        Return
            m (List(Tensor)): [N x F x T, ...] or
            s (List(Tensor)): [N x S, ...]
        """
        self.check_args(mix, training=True, valid_dim=[2, 3])
        return self._forward(mix, mode=self.training_mode)
