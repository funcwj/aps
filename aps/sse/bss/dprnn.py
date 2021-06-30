#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, List, Union

from aps.sse.bss.tcn import normalize_layer
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


class DpB(nn.Module):
    """
    DP block
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bi_inter: bool = True) -> None:
        super(DpB, self).__init__()
        self.inter_rnn = nn.LSTM(input_size,
                                 hidden_size,
                                 1,
                                 batch_first=True,
                                 bidirectional=bi_inter)
        self.inter_proj = nn.Linear(
            hidden_size * 2 if bi_inter else hidden_size, input_size)
        self.inter_norm = nn.LayerNorm(input_size)

        self.intra_rnn = nn.LSTM(input_size,
                                 hidden_size,
                                 1,
                                 batch_first=True,
                                 bidirectional=True)
        self.intra_proj = nn.Linear(hidden_size * 2, input_size)
        self.intra_norm = nn.LayerNorm(input_size)

    def _intra(self, chunk: th.Tensor) -> th.Tensor:
        """
        Go through intra block
        """
        N, F, K, L = chunk.shape
        # N x L x K x F
        chunk = chunk.permute(0, -1, 2, 1).contiguous()
        # NL x K x F
        intra_inp = chunk.view(-1, K, F)
        # NL x K x H
        intra_out, _ = self.intra_rnn(intra_inp)
        # NL x K x F
        intra_out = self.intra_norm(self.intra_proj(intra_out))
        # NL x K x F
        intra_out = intra_out + intra_inp
        # N x L x K x F
        return intra_out.view(N, L, K, F)

    def _inter(self, chunk: th.Tensor) -> th.Tensor:
        """
        Go through inter block
        """
        N, L, K, F = chunk.shape
        # N x K x L x F
        chunk = chunk.transpose(1, 2).contiguous()
        # NK x L x F
        inter_inp = chunk.view(-1, L, F)
        # NK x L x H
        inter_out, _ = self.inter_rnn(inter_inp)
        # NK x L x F
        inter_out = self.inter_norm(self.inter_proj(inter_out))
        inter_out = inter_out + inter_inp
        # N x K x L x F
        return inter_out.view(N, K, L, F)

    def forward(self, chunk: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): N x F x K x L
        Return:
            ...
        """
        if chunk.dim() not in [3, 4]:
            raise RuntimeError(
                f"DpB expects 3/4D tensor, got {chunk.dim()} instead")
        if chunk.dim() == 3:
            chunk = chunk[None, ...]
        # N x L x K x F
        intra_out = self._intra(chunk)
        # N x K x L x F
        inter_out = self._inter(intra_out)
        # N x F x K x L
        return inter_out.permute(0, -1, 1, 2)


class McB(nn.Module):
    """
    The multiply and concat (MULCAT) block
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bi_inter: bool = True) -> None:
        super(McB, self).__init__()
        self.inter_rnn1 = nn.LSTM(input_size,
                                  hidden_size,
                                  1,
                                  batch_first=True,
                                  bidirectional=bi_inter)
        self.inter_rnn2 = nn.LSTM(input_size,
                                  hidden_size,
                                  1,
                                  batch_first=True,
                                  bidirectional=bi_inter)
        self.inter_proj = nn.Linear(hidden_size * 2 +
                                    input_size if bi_inter else hidden_size +
                                    input_size,
                                    input_size,
                                    bias=False)
        self.intra_rnn1 = nn.LSTM(input_size,
                                  hidden_size,
                                  1,
                                  batch_first=True,
                                  bidirectional=True)
        self.intra_rnn2 = nn.LSTM(input_size,
                                  hidden_size,
                                  1,
                                  batch_first=True,
                                  bidirectional=True)
        self.intra_proj = nn.Linear(hidden_size * 2 + input_size,
                                    input_size,
                                    bias=False)

    def _intra(self, chunk: th.Tensor) -> th.Tensor:
        """
        Go through intra block
        """
        N, F, K, L = chunk.shape
        # N x L x K x F
        chunk = chunk.permute(0, -1, 2, 1).contiguous()
        # NL x K x F
        intra_inp = chunk.view(-1, K, F)
        rnn1, _ = self.intra_rnn1(intra_inp)
        rnn2, _ = self.intra_rnn2(intra_inp)
        # NL x K x (H+F)
        intra_cat = th.cat([rnn1 * rnn2, intra_inp], dim=-1)
        # NL x K x F
        intra_out = self.intra_proj(intra_cat)
        # NL x K x F
        intra_out = intra_out + intra_inp
        # N x L x K x F
        return intra_out.view(N, L, K, F)

    def _inter(self, chunk: th.Tensor) -> th.Tensor:
        """
        Go through inter block
        """
        N, L, K, F = chunk.shape
        # N x K x L x F
        chunk = chunk.transpose(1, 2).contiguous()
        # NK x L x F
        inter_inp = chunk.view(-1, L, F)
        rnn1, _ = self.inter_rnn1(inter_inp)
        rnn2, _ = self.inter_rnn2(inter_inp)
        # NK x L x (H+F)
        inter_cat = th.cat([rnn1 * rnn2, inter_inp], dim=-1)
        # NK x L x F
        inter_out = self.inter_proj(inter_cat)
        inter_out = inter_out + inter_inp
        # N x K x L x F
        return inter_out.view(N, K, L, F)

    def forward(self, chunk: th.Tensor) -> th.Tensor:
        """
        Args:
            chunk (Tensor): N x F x K x L
        Return:
            ...
        """
        if chunk.dim() not in [3, 4]:
            raise RuntimeError(
                f"DpB expects 3/4D tensor, got {chunk.dim()} instead")
        if chunk.dim() == 3:
            chunk = chunk[None, ...]
        # N x L x K x F
        intra_out = self._intra(chunk)
        # N x K x L x F
        inter_out = self._inter(intra_out)
        # N x F x K x L
        return inter_out.permute(0, -1, 1, 2)


class DPRNN(nn.Module):
    """
    For DPRNN parts
    """

    def __init__(self,
                 num_branch: int,
                 chunk_len: int,
                 input_norm: str = "cLN",
                 block_type: str = "dp",
                 conv_filters: int = 64,
                 proj_filters: int = 128,
                 num_layers: int = 6,
                 rnn_hidden: int = 128,
                 rnn_bi_inter: bool = True,
                 output_non_linear: str = "sigmoid") -> None:
        super(DPRNN, self).__init__()
        if block_type not in ["dp", "mc"]:
            raise RuntimeError(f"Unsupported DPRNN block: {block_type}")
        BLOCK = {"dp": DpB, "mc": McB}[block_type]
        self.non_linear = MaskNonLinear(
            output_non_linear, enable="common") if output_non_linear else None
        self.dprnn = nn.Sequential(*[
            BLOCK(proj_filters, rnn_hidden, bi_inter=rnn_bi_inter)
            for _ in range(num_layers)
        ])
        self.norm = normalize_layer(input_norm,
                                    conv_filters) if input_norm else None
        self.proj = nn.Conv1d(conv_filters, proj_filters, 1)
        # NOTE: add prelu here
        self.mask = nn.Sequential(
            nn.PReLU(), nn.Conv2d(proj_filters, num_branch * conv_filters, 1))
        self.chunk_hop, self.chunk_len = chunk_len // 2, chunk_len
        self.num_branch = num_branch

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x F x T
        Return:
            masks: N x S x F x T
        """
        N, F, T = inp.shape
        # proj + norm
        if self.norm:
            inp = self.norm(inp)
        inp = self.proj(inp)
        # N x F x T x 1 => N x FK x L
        rnn_inp = tf.unfold(inp[..., None], (self.chunk_len, 1),
                            stride=self.chunk_hop)
        L = rnn_inp.shape[-1]
        # N x F x K x L
        rnn_inp = rnn_inp.view(N, inp.shape[1], self.chunk_len, L)
        # N x F x K x L
        rnn_out = self.dprnn(rnn_inp)
        # N x SF x K x L
        rnn_out = self.mask(rnn_out).contiguous()
        # NS x FK x L
        rnn_out = rnn_out.view(N * self.num_branch, -1, L)
        # NS x F x T x 1
        masks = tf.fold(rnn_out, (T, 1), (self.chunk_len, 1),
                        stride=self.chunk_hop)
        # N x S x F x T
        masks = masks.view(N, self.num_branch, F, -1)
        if self.non_linear:
            return self.non_linear(masks)
        else:
            return masks


@ApsRegisters.sse.register("sse@time_dprnn")
class TimeDPRNN(SseBase):
    """
    Time domain DP (dual-path) RNN
    """

    def __init__(self,
                 num_spks: int = 2,
                 input_norm: str = "cLN",
                 block_type: str = "dp",
                 conv_kernels: int = 16,
                 conv_filters: int = 64,
                 proj_filters: int = 128,
                 chunk_len: int = 100,
                 num_layers: int = 6,
                 rnn_bi_inter: bool = True,
                 rnn_hidden: int = 128,
                 non_linear: str = "relu",
                 masking: bool = True) -> None:
        super(TimeDPRNN, self).__init__(None, training_mode="time")
        # conv1d encoder
        self.encoder = nn.Conv1d(1,
                                 conv_filters,
                                 conv_kernels,
                                 stride=conv_kernels // 2,
                                 bias=False,
                                 padding=0)
        # conv1d decoder
        self.decoder = nn.ConvTranspose1d(conv_filters,
                                          1,
                                          conv_kernels,
                                          stride=conv_kernels // 2,
                                          bias=False,
                                          padding=0)
        self.dprnn = DPRNN(num_spks,
                           chunk_len,
                           input_norm=input_norm,
                           block_type=block_type,
                           conv_filters=conv_filters,
                           proj_filters=proj_filters,
                           num_layers=num_layers,
                           rnn_hidden=rnn_hidden,
                           rnn_bi_inter=rnn_bi_inter,
                           output_non_linear=non_linear)
        self.masking = masking
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
        w = tf.relu(self.encoder(mix[:, None, :]))
        # N x S x F x T
        masks = self.dprnn(w)
        # masking or not
        if not self.masking:
            w = 1
        if self.num_spks == 1:
            return self.decoder(masks[:, 0] * w)[:, 0]
        else:
            return [
                self.decoder(masks[:, s] * w)[:, 0]
                for s in range(self.num_spks)
            ]


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
                 input_norm: str = "",
                 block_type: str = "dp",
                 proj_filters: int = 256,
                 chunk_len: int = 64,
                 num_layers: int = 6,
                 rnn_hidden: int = 256,
                 rnn_bi_inter: bool = True,
                 training_mode: str = "freq") -> None:
        super(FreqDPRNN, self).__init__(enh_transform,
                                        training_mode=training_mode)
        self.dprnn = DPRNN(num_spks,
                           chunk_len,
                           input_norm=input_norm,
                           block_type=block_type,
                           conv_filters=num_bins,
                           proj_filters=proj_filters,
                           num_layers=num_layers,
                           rnn_hidden=rnn_hidden,
                           rnn_bi_inter=rnn_bi_inter,
                           output_non_linear=non_linear)
        assert enh_transform is not None
        self.num_spks = num_spks

    def _forward(self, mix: th.Tensor,
                 mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
        """
        # mix_feat: N x T x F
        # mix_stft: N x (C) x F x T
        mix_feat, mix_stft, _ = self.enh_transform(mix, None)
        if mix_stft.dim() == 4:
            # N x F x T
            mix_stft = mix_stft[:, 0]
        # N x F x T
        w = th.transpose(mix_feat, 1, 2)
        # N x 2 x F x T
        masks = self.dprnn(w)
        if self.num_spks == 1:
            masks = masks[:, 0]
        else:
            masks = [masks[:, s] for s in range(self.num_spks)]
        # N x F x T, ...
        if mode == "freq":
            return masks
        else:
            decoder = self.enh_transform.inverse_stft
            if self.num_spks == 1:
                enh_stft = mix_stft * masks
                enh = decoder((enh_stft.real, enh_stft.imag), input="complex")
            else:
                enh_stft = [mix_stft * m for m in masks]
                enh = [
                    decoder((s.real, s.imag), input="complex") for s in enh_stft
                ]
            return enh

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
