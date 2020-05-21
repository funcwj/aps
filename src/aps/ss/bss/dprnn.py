#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as tf


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
                f"Expect 3/4D tensor, got {chunk.dim()} instead")
        if chunk.dim() == 3:
            chunk = chunk[None, ...]
        print(chunk.shape)
        # N x L x K x F
        intra_out = self._intra(chunk)
        # N x K x L x F
        inter_out = self._inter(intra_out)
        # N x F x K x L
        return inter_out.permute(0, -1, 1, 2)


class DPRNN(nn.Module):
    """
    DP (dual-path) RNN
    """
    def __init__(self,
                 num_spks=2,
                 conv_kernels=16,
                 conv_filters=64,
                 chunk_len=100,
                 dprnn_layers=6,
                 dprnn_bi_inter=True,
                 dprnn_hidden=128):
        super(DPRNN, self).__init__()
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
        self.mask = nn.Conv2d(conv_filters, 2 * conv_filters, 1)
        self.dprnn = nn.Sequential(*[
            DpB(conv_filters, dprnn_hidden, bi_inter=dprnn_bi_inter)
            for _ in range(dprnn_layers)
        ])
        self.num_spks = num_spks
        self.chunk_hop, self.chunk_len = chunk_len // 2, chunk_len

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        if s.dim() not in [1, 2]:
            raise RuntimeError(f"Expect 1/2D tensor, but got {s.dim()}")
        if s.dim() == 1:
            s = s[None, ...]
        # N x 1 x S => N x F x T
        w = tf.relu(self.encoder(s[:, None, :]))
        N, F, T = w.shape
        # N x F x T x 1 => N x FK x L
        rnn_inp = tf.unfold(w[..., None], (self.chunk_len, 1),
                            stride=self.chunk_hop)
        L = rnn_inp.shape[-1]
        # N x F x K x L
        rnn_inp = rnn_inp.view(N, F, self.chunk_len, L)
        # N x F x K x L
        rnn_out = self.dprnn(rnn_inp)
        # N x 2F x K x L
        rnn_out = self.mask(rnn_out)
        # N2 x FK x L
        rnn_out = rnn_out.view(N * self.num_spks, -1, L)
        # N2 x F x T x 1
        masks = tf.fold(rnn_out, (T, 1), (self.chunk_len, 1),
                        stride=self.chunk_hop)
        # N x 2 x F x T
        masks = masks.view(N, self.num_spks, F, -1)
        return [self.decoder(masks[:, s] * w) for s in range(self.num_spks)]


def run():
    dprnn = DpRNN(num_spks=2, chunk_len=100)
    print(dprnn)
    x = th.rand(2, 320000)
    m = dprnn(x)
    print(m[0].shape)


if __name__ == "__main__":
    run()