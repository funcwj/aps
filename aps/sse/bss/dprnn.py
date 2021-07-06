#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, List, Union

from aps.transform.asr import TFTransposeTransform
from aps.sse.bss.tcn import normalize_layer
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


class LSTMBlock(nn.Module):
    """
    LSTM layer in DP-RNN
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bidirectional: bool = True) -> None:
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            1,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.proj = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                              input_size)
        self.norm = nn.LayerNorm(input_size)

    def forward(self, chunk: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): N x K x L x C
        Return:
            chunk (Tensor): N x L x K x C
        """
        N, K, L, C = chunk.shape
        # N x L x K x C
        chunk = chunk.transpose(1, 2).contiguous()
        # NL x K x C
        rnn_inp = chunk.view(-1, K, C)
        # NL x K x H
        rnn_out, _ = self.lstm(rnn_inp)
        # NL x K x C
        chunk = rnn_inp + self.norm(self.proj(rnn_out))
        # N x L x K x C
        return chunk.view(N, L, K, -1)


class DPRNN(nn.Module):
    """
    For DPRNN parts
    """

    def __init__(self,
                 num_bins: int = 256,
                 num_spks: int = 2,
                 num_layers: int = 2,
                 chunk_size: int = 320,
                 rnn_hidden: int = 128,
                 bidirectional: bool = True) -> None:
        super(DPRNN, self).__init__()
        self.chunk_size = chunk_size
        # [intra, inter, intra, inter, ...]
        separator = [
            LSTMBlock(num_bins,
                      rnn_hidden,
                      bidirectional=True if i % 2 == 0 else bidirectional)
            for i in range(num_layers * 2)
        ]
        self.separator = nn.Sequential(*separator)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x F x T
        Return:
            masks (Tensor): N x S*F x T
        """
        batch_size, num_bins, num_frames = inp.shape
        # N x C x T x 1 => N x CK x L
        chunks = tf.unfold(inp[..., None], (self.chunk_size, 1),
                           stride=self.chunk_size // 2)
        # N x C x K x L
        chunks = chunks.view(batch_size, num_bins, self.chunk_size, -1)
        # N x K x L x C
        chunks = chunks.permute(0, 2, 3, 1)
        # N x K x L x C
        chunks = self.separator(chunks)
        # N x C x K x L
        chunks = chunks.permute(0, 3, 1, 2)
        # N x CK x L
        chunks = chunks.contiguous()
        chunks = chunks.view(batch_size, -1, chunks.shape[-1])
        # N x C x T x 1
        out = tf.fold(chunks, (num_frames, 1), (self.chunk_size, 1),
                      stride=self.chunk_size // 2)
        # N x C*S x T
        return out[..., 0]


@ApsRegisters.sse.register("sse@time_dprnn")
class TimeDPRNN(SseBase):
    """
    Time domain DP (dual-path) RNN
    """

    def __init__(self,
                 num_spks: int = 2,
                 num_bins: int = 64,
                 stride: int = 8,
                 kernel: int = 16,
                 chunk_size: int = 100,
                 num_layers: int = 6,
                 bidirectional: bool = True,
                 rnn_hidden: int = 128,
                 non_linear: str = "relu") -> None:
        super(TimeDPRNN, self).__init__(None, training_mode="time")
        # conv1d encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, num_bins, kernel_size=kernel, stride=stride,
                      padding=0), nn.ReLU(), normalize_layer("cLN", num_bins))
        self.separator = DPRNN(num_bins=num_bins,
                               num_spks=num_spks,
                               num_layers=num_layers,
                               chunk_size=chunk_size,
                               rnn_hidden=rnn_hidden,
                               bidirectional=bidirectional)
        self.mask = nn.Sequential(
            nn.Conv1d(num_bins, num_bins * num_spks, 1),
            MaskNonLinear(non_linear, enable="positive_wo_softmax"))
        # conv1d decoder
        self.decoder = nn.ConvTranspose1d(num_bins,
                                          1,
                                          kernel_size=kernel,
                                          stride=stride,
                                          bias=True)
        self.num_spks = num_spks

    def infer(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            [Tensor, ...]: S
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
        # N x 1 x S => N x F x T
        w = self.encoder(mix[:, None, :])
        # N x S*F x T
        mask = self.mask(self.separator(w))
        # [N x C x T, ...]
        m = th.chunk(mask, self.num_spks, 1)
        # S x N x C x T
        m = th.stack(m, dim=0)
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        bss = [self.decoder(x)[:, 0] for x in s]
        return bss[0] if self.num_spks == 1 else bss


@ApsRegisters.sse.register("sse@freq_dprnn")
class FreqDPRNN(SseBase):
    """
    Frequency domain DP (dual-path) RNN
    """

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 num_spks: int = 2,
                 num_bins: int = 257,
                 non_linear: str = "relu",
                 chunk_size: int = 64,
                 num_layers: int = 6,
                 rnn_hidden: int = 256,
                 bidirectional: bool = True,
                 training_mode: str = "freq") -> None:
        super(FreqDPRNN, self).__init__(enh_transform,
                                        training_mode=training_mode)
        assert enh_transform is not None
        self.swap = TFTransposeTransform()
        self.separator = DPRNN(num_bins=num_bins,
                               num_spks=num_spks,
                               num_layers=num_layers,
                               chunk_size=chunk_size,
                               rnn_hidden=rnn_hidden,
                               bidirectional=bidirectional)
        self.mask = nn.Sequential(nn.Conv1d(num_bins, num_bins * num_spks, 1),
                                  MaskNonLinear(non_linear, enable="common"))
        self.num_spks = num_spks

    def _forward(self, mix: th.Tensor,
                 mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
        """
        # mix_stft: N x x F x T
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
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            ret = self._forward(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        return self._forward(mix, self.training_mode)
