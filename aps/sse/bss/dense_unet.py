#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, List, Union, Tuple
from aps.sse.bss.dccrn import LSTMWrapper, parse_1dstr, parse_2dstr
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters
"""
UNet used in Wang's paper
"""
ComplexPair = Tuple[th.Tensor, th.Tensor]
DenseUnetRetType = Union[th.Tensor, List[th.Tensor], List[ComplexPair]]


class EncoderBlock(nn.Module):
    """
    Conv2d block in encoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: str = 1,
                 padding: Tuple[int, int] = (1, 1),
                 dropout: float = 0,
                 norm: str = "IN",
                 first_layer: bool = False) -> None:
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

    def forward(self, inp: th.Tensor) -> th.Tensor:
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
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: str = 1,
                 padding: Tuple[int, int] = (1, 1),
                 output_padding: Tuple[int, int] = (0, 0),
                 dropout: float = 0,
                 norm: str = "IN",
                 last_layer: bool = False) -> None:
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

    def forward(self, inp: th.Tensor) -> th.Tensor:
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
                 in_channels: int,
                 out_channels: int,
                 growth_rate: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 num_layers: int = 5,
                 stride: int = 1,
                 norm: str = "IN") -> None:
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

    def forward(self, inp: th.Tensor) -> th.Tensor:
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
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: int = 1,
                 dropout: float = 0,
                 padding: Tuple[int, int] = (1, 1),
                 norm: str = "IN",
                 inner_dense_layer: int = 5,
                 first_layer: bool = False) -> None:
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

    def forward(self, inp: th.Tensor) -> th.Tensor:
        sub1 = self.sub1(inp)
        return self.sub2(sub1)


class DecoderDenseBlock(nn.Module):
    """
    Decoder block + Dense block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: int = 1,
                 padding: Tuple[int, int] = (1, 1),
                 output_padding: Tuple[int, int] = (0, 0),
                 dropout: float = 0,
                 norm: str = "IN",
                 inner_dense_layer: int = 5,
                 last_layer: bool = False,
                 last_out_channels: int = 2) -> None:
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

    def forward(self, inp: th.Tensor) -> th.Tensor:
        sub1 = self.sub1(inp)
        return self.sub2(sub1)


class Encoder(nn.Module):
    """
    Encoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 num_dense_blocks: int = 4,
                 dropout: float = 0,
                 norm: str = "IN") -> None:
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

    def forward(self, x: th.Tensor) -> List[th.Tensor]:
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
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 O: List[int],
                 enc_channel: Optional[List[int]] = None,
                 dropout: float = 0,
                 norm: str = "IN",
                 num_dense_blocks: int = 4) -> None:
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

    def forward(self, x: th.Tensor, enc_h: List[th.Tensor]) -> th.Tensor:
        # N = len(self.decoders)
        for index, conv in enumerate(self.decoders):
            if index == 0:
                x = conv(x)
            else:
                x = th.cat([x, enc_h[index - 1]], 1)
                x = conv(x)
            # print(f"encoder-{N - 1 - index}: {x.shape}")
        return x


@ApsRegisters.sse.register("sse@dense_unet")
class DenseUnet(SseBase):
    """
    Boosted Unet proposed by Wang
    """

    def __init__(self,
                 inp_cplx: bool = False,
                 out_cplx: bool = False,
                 K: str = "3,3;3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S: str = "1,1;2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 P: str = "0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1",
                 O: str = "0,0,0,0,0,0,0,0",
                 enc_channel: str = "16,32,32,32,32,64,128,384",
                 dec_channel: str = "32,16,32,32,32,32,64,128",
                 conv_dropout: float = 0,
                 norm: str = "IN",
                 num_spks: int = 2,
                 rnn_hidden: int = 512,
                 rnn_layers: int = 2,
                 rnn_resize: int = 512,
                 rnn_bidir: bool = False,
                 rnn_dropout: float = 0,
                 num_dense_blocks: int = 4,
                 enh_transform: Optional[nn.Module] = None,
                 non_linear: str = "sigmoid",
                 non_linear_scale: int = 1,
                 non_linear_clip: Optional[float] = None,
                 training_mode: str = "freq") -> None:
        super(DenseUnet, self).__init__(enh_transform,
                                        training_mode=training_mode)
        assert enh_transform is not None
        if non_linear:
            self.non_linear = MaskNonLinear(non_linear,
                                            enable="all_wo_softmax",
                                            scale=non_linear_scale,
                                            value_clip=non_linear_clip)
        else:
            # complex mapping
            self.non_linear = None
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
        self.inp_cplx = inp_cplx
        self.out_cplx = out_cplx

    def sep(self,
            m: th.Tensor,
            sr: th.Tensor,
            si: th.Tensor,
            mode: str = "freq") -> Union[th.Tensor, ComplexPair]:
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

    def infer(self, mix: th.Tensor, mode: str = "time") -> DenseUnetRetType:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            sep = self._forward(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                if isinstance(sep[0], th.Tensor):
                    bss = [s[0] for s in sep]
                else:
                    bss = [(s[0][0], s[1][0]) for s in sep]
                return bss

    def _forward(self, mix: th.Tensor, mode: str = "freq") -> DenseUnetRetType:
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

    def forward(self, s: th.Tensor) -> DenseUnetRetType:
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True, valid_dim=[2])
        return self._forward(s, mode=self.training_mode)
