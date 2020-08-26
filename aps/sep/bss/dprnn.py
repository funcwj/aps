#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from .tasnet import build_norm


class DpB(nn.Module):
    """
    DP block
    """
    def __init__(self, input_size, hidden_size, bi_inter=True):
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

    def _intra(self, chunk):
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

    def _inter(self, chunk):
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

    def forward(self, chunk):
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
    def __init__(self, input_size, hidden_size, bi_inter=True):
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

    def _intra(self, chunk):
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

    def _inter(self, chunk):
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

    def forward(self, chunk):
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
                 num_spks=2,
                 chunk_len=100,
                 input_norm="cLN",
                 conv_filters=64,
                 proj_filters=128,
                 dprnn_layers=6,
                 dprnn_hidden=128,
                 dprnn_bi_inter=True,
                 dprnn_block="dp",
                 output_non_linear="sigmoid"):
        super(DPRNN, self).__init__()
        supported_nonlinear = {
            "relu": tf.relu,
            "sigmoid": th.sigmoid,
            "": None
        }
        if output_non_linear not in supported_nonlinear:
            raise RuntimeError(
                f"Unsupported non-linear function: {output_non_linear}")
        if dprnn_block not in ["dp", "mc"]:
            raise RuntimeError(f"Unsupported DPRNN block: {dprnn_block}")
        BLOCK = {"dp": DpB, "mc": McB}[dprnn_block]
        self.non_linear = supported_nonlinear[output_non_linear]
        self.dprnn = nn.Sequential(*[
            BLOCK(proj_filters, dprnn_hidden, bi_inter=dprnn_bi_inter)
            for _ in range(dprnn_layers)
        ])
        self.norm = build_norm(input_norm,
                               conv_filters) if input_norm else None
        self.proj = nn.Conv1d(conv_filters, proj_filters, 1)
        # NOTE: add prelu here
        self.mask = nn.Sequential(
            nn.PReLU(), nn.Conv2d(proj_filters, num_spks * conv_filters, 1))
        self.chunk_hop, self.chunk_len = chunk_len // 2, chunk_len
        self.num_spks = num_spks

    def check_args(self, mix, training=True):
        """
        Check input arguments
        """
        if not training and mix.dim() != 1:
            raise RuntimeError(
                f"DPRNN expects 1D tensor (inference), but got {mix.dim()}")
        if training and mix.dim() != 2:
            raise RuntimeError(
                f"DPRNN expects 2D tensor (training), but got {mix.dim()}")

    def forward(self, inp):
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
        rnn_out = rnn_out.view(N * self.num_spks, -1, L)
        # NS x F x T x 1
        masks = tf.fold(rnn_out, (T, 1), (self.chunk_len, 1),
                        stride=self.chunk_hop)
        # N x S x F x T
        masks = masks.view(N, self.num_spks, F, -1)
        if self.non_linear:
            return self.non_linear(masks)
        else:
            return masks


class TimeDPRNN(DPRNN):
    """
    Time domain DP (dual-path) RNN
    """
    def __init__(self,
                 num_spks=2,
                 input_norm="cLN",
                 conv_kernels=16,
                 conv_filters=64,
                 proj_filters=128,
                 chunk_len=100,
                 dprnn_layers=6,
                 dprnn_bi_inter=True,
                 dprnn_hidden=128,
                 dprnn_block="dp",
                 non_linear="relu",
                 masking=True):
        super(TimeDPRNN, self).__init__(num_spks=num_spks,
                                        chunk_len=chunk_len,
                                        input_norm=input_norm,
                                        conv_filters=conv_filters,
                                        proj_filters=proj_filters,
                                        dprnn_layers=dprnn_layers,
                                        dprnn_hidden=dprnn_hidden,
                                        dprnn_block=dprnn_block,
                                        dprnn_bi_inter=dprnn_bi_inter,
                                        output_non_linear=non_linear)
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
        self.masking = masking

    def infer(self, mix):
        """
        Args:
            mix (Tensor): S
        Return:
            [Tensor, ...]: S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, ...]
            sep = self.forward(mix)
            return sep[0] if self.num_spks == 1 else [s[0] for s in sep]

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True)
        # N x 1 x S => N x F x T
        w = tf.relu(self.encoder(mix[:, None, :]))
        # N x S x F x T
        masks = super().forward(w)
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


class FreqDPRNN(DPRNN):
    """
    Frequency domain DP (dual-path) RNN
    """
    def __init__(self,
                 enh_transform=None,
                 num_spks=2,
                 num_bins=257,
                 non_linear="relu",
                 input_norm="",
                 proj_filters=256,
                 chunk_len=64,
                 dprnn_layers=6,
                 dprnn_bi_inter=True,
                 dprnn_hidden=128,
                 dprnn_block="dp",
                 training_mode="freq"):
        super(FreqDPRNN, self).__init__(num_spks=num_spks,
                                        chunk_len=chunk_len,
                                        input_norm=input_norm,
                                        conv_filters=num_bins,
                                        proj_filters=proj_filters,
                                        dprnn_layers=dprnn_layers,
                                        dprnn_hidden=dprnn_hidden,
                                        dprnn_block=dprnn_block,
                                        dprnn_bi_inter=dprnn_bi_inter,
                                        output_non_linear=non_linear)
        if enh_transform is None:
            raise RuntimeError("FreqDPRNN: enh_transform can not be None")
        self.enh_transform = enh_transform
        self.mode = training_mode

    def _forward(self, mix, mode):
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
        masks = super().forward(w)
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
                    decoder((s.real, s.imag), input="complex")
                    for s in enh_stft
                ]
            return enh

    def infer(self, mix, mode="time"):
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            ret = self._forward(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True)
        return self._forward(mix, self.mode)