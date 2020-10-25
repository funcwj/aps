#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.sep.utils import MaskNonLinear
from .dccrn import LSTMWrapper, parse_1dstr, parse_2dstr
"""
UNet used in Wang's paper
"""


class EncoderBlock(nn.Module):
    """
    Conv2d block in encoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=(1, 1),
                 dropout=0,
                 norm="IN",
                 first_layer=False):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)
        NormLayer = nn.InstanceNorm2d if norm == "IN" else nn.BatchNorm2d
        if not first_layer:
            self.elu = nn.ELU()
            self.inst = NormLayer(out_channels)
            self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        else:
            self.elu, self.inst = None, None

    def forward(self, inp):
        out = self.conv(inp)
        if self.inst is None:
            return out
        out = self.elu(out)
        if self.dropout:
            out = self.dropout(out)
        return self.inst(out)


class DecoderBlock(nn.Module):
    """
    Deconv2d block in dncoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=(1, 1),
                 output_padding=(0, 0),
                 norm="IN",
                 dropout=0,
                 last_layer=False):
        super(DecoderBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       output_padding=output_padding,
                                       padding=padding)
        NormLayer = nn.InstanceNorm2d if norm == "IN" else nn.BatchNorm2d
        if not last_layer:
            self.elu = nn.ELU()
            self.inst = NormLayer(out_channels)
            self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else None
        else:
            self.elu, self.inst = None, None

    def forward(self, inp):
        out = self.conv(inp)
        if self.inst is None:
            return out
        out = self.elu(out)
        if self.dropout:
            out = self.dropout(out)
        return self.inst(out)


class DenseBlock(nn.Module):
    """
    Dense block in encoder/decoder
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 growth_rate,
                 kernel_size=(3, 3),
                 num_layers=5,
                 stride=1,
                 norm="IN"):
        super(DenseBlock, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(in_channels if i == 0 else in_channels +
                         growth_rate * i,
                         growth_rate if i != num_layers - 1 else out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         norm=norm,
                         padding=(1, 1)) for i in range(num_layers)
        ])

    def forward(self, inp):
        inputs = [inp]
        for conv in self.blocks:
            # inp = th.cat(inputs, dim=1)
            inp = conv(th.cat(inputs, dim=1))
            inputs.append(inp)
        return inp


class EncoderDenseBlock(nn.Module):
    """
    Encoder block + Dense block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 dropout=0,
                 padding=(1, 1),
                 norm="IN",
                 inner_dense_layer=5,
                 first_layer=False):
        super(EncoderDenseBlock, self).__init__()
        self.sub1 = EncoderBlock(in_channels,
                                 out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dropout=dropout,
                                 norm=norm,
                                 first_layer=first_layer)
        self.sub2 = DenseBlock(out_channels,
                               out_channels,
                               out_channels,
                               kernel_size=(3, 3),
                               num_layers=inner_dense_layer,
                               stride=(1, 1),
                               norm=norm)

    def forward(self, inp):
        sub1 = self.sub1(inp)
        return self.sub2(sub1)


class DecoderDenseBlock(nn.Module):
    """
    Decoder block + Dense block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=(1, 1),
                 output_padding=(0, 0),
                 dropout=0,
                 norm="IN",
                 inner_dense_layer=5,
                 last_layer=False,
                 last_out_channels=2):
        super(DecoderDenseBlock, self).__init__()
        self.sub1 = DenseBlock(in_channels * 2,
                               in_channels * 2,
                               in_channels,
                               kernel_size=(3, 3),
                               num_layers=inner_dense_layer,
                               stride=(1, 1),
                               norm=norm)
        self.sub2 = DecoderBlock(
            in_channels * 2,
            last_out_channels if last_layer else out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dropout=dropout,
            output_padding=output_padding,
            norm=norm,
            last_layer=last_layer)

    def forward(self, inp):
        sub1 = self.sub1(inp)
        return self.sub2(sub1)


class Encoder(nn.Module):
    """
    Encoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self, K, S, C, P, num_dense_blocks=4, dropout=0, norm="IN"):
        super(Encoder, self).__init__()
        total_layers = len(C) - 1
        layers = [
            EncoderDenseBlock(C[i],
                              C[i + 1],
                              kernel_size=K[i],
                              padding=P[i],
                              stride=S[i],
                              dropout=dropout,
                              norm=norm,
                              first_layer=(i == 0))
            for i in range(num_dense_blocks)
        ]
        layers += [
            EncoderBlock(C[i],
                         C[i + 1],
                         kernel_size=K[i],
                         stride=S[i],
                         padding=P[i],
                         dropout=dropout,
                         norm=norm,
                         first_layer=(i == 0))
            for i in range(num_dense_blocks, total_layers)
        ]
        self.encoders = nn.ModuleList(layers)

    def forward(self, x):
        enc_h = []
        for index, conv in enumerate(self.encoders):
            x = conv(x)
            # print(f"encoder-{index}: {x.shape}")
            enc_h.append(x)
        return enc_h


class Decoder(nn.Module):
    """
    Decoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 K,
                 S,
                 C,
                 P,
                 O,
                 enc_channel=None,
                 dropout=0,
                 norm="IN",
                 num_dense_blocks=4):
        super(Decoder, self).__init__()
        total_layers = len(C) - 1
        layers = [
            DecoderBlock(enc_channel[i] * 2,
                         C[i],
                         kernel_size=K[i],
                         stride=S[i],
                         padding=P[i],
                         output_padding=(O[i], 0),
                         dropout=dropout,
                         norm=norm,
                         last_layer=(i == total_layers - 1))
            for i in range(total_layers - num_dense_blocks)
        ]
        layers += [
            DecoderDenseBlock(
                enc_channel[i],
                # 32 if C[i] == 64 else C[i],
                C[i],
                kernel_size=K[i],
                stride=S[i],
                padding=P[i],
                output_padding=(O[i], 0),
                dropout=dropout,
                norm=norm,
                last_out_channels=C[-1],
                last_layer=(i == total_layers - 1))
            for i in range(total_layers - num_dense_blocks, total_layers)
        ]
        self.decoders = nn.ModuleList(layers)

    def forward(self, x, enc_h):
        # N = len(self.decoders)
        for index, conv in enumerate(self.decoders):
            if index == 0:
                x = conv(x)
            else:
                x = th.cat([x, enc_h[index - 1]], 1)
                x = conv(x)
            # print(f"encoder-{N - 1 - index}: {x.shape}")
        return x


class DenseUnet(nn.Module):
    """
    Boosted Unet proposed by Wang
    """

    def __init__(self,
                 inp_cplx=False,
                 out_cplx=False,
                 K="3,3;3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S="1,1;2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 P="0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1",
                 O="0,0,0,0,0,0,0,0",
                 enc_channel="16,32,32,32,32,64,128,384",
                 dec_channel="32,16,32,32,32,32,64,128",
                 conv_dropout=0,
                 norm="IN",
                 num_spks=2,
                 rnn_hidden=512,
                 rnn_layers=2,
                 rnn_resize=512,
                 rnn_bidir=False,
                 rnn_dropout=0,
                 num_dense_blocks=4,
                 enh_transform=None,
                 non_linear="sigmoid",
                 non_linear_scale=1,
                 non_linear_clip=None,
                 training_mode="freq"):
        super(DenseUnet, self).__init__()
        if enh_transform is None:
            raise RuntimeError("Missing configuration for enh_transform")
        if non_linear:
            self.non_linear = MaskNonLinear(non_linear,
                                            scale=non_linear_scale,
                                            clip=non_linear_clip)
        else:
            # complex mapping
            self.non_linear = None
        self.enh_transform = enh_transform
        K = parse_2dstr(K)
        # make sure stride size on time axis is 1
        S = parse_2dstr(S)
        P = parse_2dstr(P)
        O = parse_1dstr(O)
        enc_channel = parse_1dstr(enc_channel)
        dec_channel = parse_1dstr(dec_channel)
        self.encoder = Encoder(K,
                               S, [3 if inp_cplx else 1] + enc_channel,
                               P,
                               dropout=conv_dropout,
                               num_dense_blocks=num_dense_blocks)
        self.decoder = Decoder(K[::-1],
                               S[::-1],
                               dec_channel[::-1] +
                               [num_spks * (2 if out_cplx else 1)],
                               P[::-1],
                               O[::-1],
                               enc_channel=enc_channel[::-1],
                               dropout=conv_dropout,
                               num_dense_blocks=num_dense_blocks)
        self.rnn = LSTMWrapper(rnn_resize,
                               hidden_size=rnn_hidden,
                               cplx=False,
                               dropout=rnn_dropout,
                               num_layers=rnn_layers,
                               bidirectional=rnn_bidir)
        self.num_spks = num_spks
        self.mode = training_mode
        self.inp_cplx = inp_cplx
        self.out_cplx = out_cplx

    def sep(self, m, sr, si, mode="freq"):
        decoder = self.enh_transform.inverse_stft
        # m: N x 2 x F x T
        if self.out_cplx:
            # N x F x T
            mr, mi = m[:, 0], m[:, 1]
            # use mask
            if self.non_linear:
                m_abs = (mr**2 + mi**2)**0.5
                m_mag = self.non_linear(m_abs)
                if mode == "freq":
                    # s = (si**2 + sr**2)**0.5 * m_mag
                    s = m_mag
                else:
                    mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
                    s = decoder((sr * mr - si * mi, sr * mi + si * mr),
                                input="complex")
            # use mapping
            else:
                if mode == "freq":
                    s = (mr, mi)
                else:
                    s = decoder((mr, mi), input="complex")
        else:
            if self.non_linear:
                m = self.non_linear(m[:, 0])
                if mode == "freq":
                    s = m
                else:
                    s = decoder((sr * m, si * m), input="complex")
            else:
                m = m[:, 0]
                if mode == "freq":
                    s = m
                else:
                    s_abs = th.sqrt(sr**2 + si**2)
                    s = decoder((m * sr / s_abs, m * si / s_abs),
                                input="complex")
        return s

    def check_args(self, mix, training=True):
        if not training and mix.dim() != 1:
            raise RuntimeError("DenseUnet expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError("DenseUnet expects 2D tensor (training), " +
                               f"got {mix.dim()} instead")

    def infer(self, mix, mode="time"):
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            sep = self._forward(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def _forward(self, mix, mode="freq"):
        # NOTE: update real input!
        if self.inp_cplx:
            # N x F x T
            sr, si = self.enh_transform.forward_stft(mix, output="complex")
            mag = th.sqrt(sr**2 + si**2)
            # N x 2 x F x T
            s = th.stack([sr, si, mag], 1)
        else:
            # N x F x T
            # s = (sr**2 + si**2)**0.5
            # NOTE: using feature instead of magnitude
            s, stft, _ = self.enh_transform(mix, None)
            # N x T x F => N x F x T
            s = s.transpose(1, 2)
            sr, si = stft.real, stft.imag
        # encoder
        if self.inp_cplx:
            enc_h = self.encoder(s)
        else:
            enc_h = self.encoder(s[:, None])
        enc_h, h = enc_h[:-1], enc_h[-1]
        out_h = self.rnn(h)
        h = th.cat([out_h, h], 1)
        # print(h.shape)
        # reverse
        enc_h = enc_h[::-1]
        # decoder
        # N x C x F x T
        spk_m = self.decoder(h, enc_h)
        if self.num_spks == 1:
            return self.sep(spk_m, sr, si, mode=mode)
        else:
            chunk_m = th.chunk(spk_m, self.num_spks, 1)
            return [self.sep(m, sr, si, mode=mode) for m in chunk_m]

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True)
        return self._forward(s, mode=self.mode)
