#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

import torch.nn.functional as F

from typing import Tuple, List, Union, Optional
from aps.sse.base import SseBase
from aps.libs import ApsRegisters


def parse_1dstr(sstr: str) -> List[int]:
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr: str) -> List[List[int]]:
    return [parse_1dstr(tok) for tok in sstr.split(";")]


class ComplexConv2d(nn.Module):
    """
    Complex 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__()
        self.real = nn.Conv2d(*args, **kwargs)
        self.imag = nn.Conv2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x C x 2F x T
        Return:
            y (Tensor): N x C' x 2F' x T'
        """
        xr, xi = th.chunk(x, 2, -2)
        yr = self.real(xr) - self.imag(xi)
        yi = self.imag(xr) + self.real(xi)
        y = th.cat([yr, yi], -2)
        return y


class ComplexConvTranspose2d(nn.Module):
    """
    Complex Transpose 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConvTranspose2d, self).__init__()
        self.real = nn.ConvTranspose2d(*args, **kwargs)
        self.imag = nn.ConvTranspose2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x C x 2F x T
        Return:
            y (Tensor): N x C' x 2F' x T'
        """
        xr, xi = th.chunk(x, 2, -2)
        yr = self.real(xr) - self.imag(xi)
        yi = self.imag(xr) + self.real(xi)
        y = th.cat([yr, yi], -2)
        return y


class ComplexBatchNorm2d(nn.Module):
    """
    A easy implementation of complex 2d batchnorm
    """

    def __init__(self, *args, **kwargs):
        super(ComplexBatchNorm2d, self).__init__()
        self.real_bn = nn.BatchNorm2d(*args, **kwargs)
        self.imag_bn = nn.BatchNorm2d(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        xr, xi = th.chunk(x, 2, -2)
        xr = self.real_bn(xr)
        xi = self.imag_bn(xi)
        x = th.cat([xr, xi], -2)
        return x


class EncoderBlock(nn.Module):
    """
    Convolutional block in encoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int],
                 stride: int = 1,
                 padding: int = 0,
                 causal: bool = False,
                 cplx: bool = True) -> None:
        super(EncoderBlock, self).__init__()
        conv_impl = ComplexConv2d if cplx else nn.Conv2d
        # NOTE: time stride should be 1
        var_kt = kernel_size[1] - 1
        time_axis_pad = var_kt if causal else var_kt // 2
        self.conv = conv_impl(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=(padding, time_axis_pad))
        if cplx:
            self.bn = ComplexBatchNorm2d(out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.causal = causal
        self.time_axis_pad = time_axis_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        x = self.conv(x)
        if self.causal:
            x = x[..., :-self.time_axis_pad]
        x = self.bn(x)
        x = F.leaky_relu(x)
        return x


class DecoderBlock(nn.Module):
    """
    Convolutional block in decoder
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int],
                 stride: int = 1,
                 padding: int = 0,
                 output_padding: int = 0,
                 causal: bool = False,
                 cplx: bool = True,
                 last_layer: bool = False) -> None:
        super(DecoderBlock, self).__init__()
        conv_impl = ComplexConvTranspose2d if cplx else nn.ConvTranspose2d
        var_kt = kernel_size[1] - 1
        time_axis_pad = var_kt if causal else var_kt // 2
        self.trans_conv = conv_impl(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=(padding, var_kt - time_axis_pad),
                                    output_padding=(output_padding, 0))
        if last_layer:
            self.bn = None
        else:
            if cplx:
                self.bn = ComplexBatchNorm2d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        self.causal = causal
        self.time_axis_pad = time_axis_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        x = self.trans_conv(x)
        if self.causal:
            x = x[..., :-self.time_axis_pad]
        if self.bn:
            x = self.bn(x)
            x = F.leaky_relu(x)
        return x


class Encoder(nn.Module):
    """
    Encoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 cplx: bool,
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 causal: bool = False) -> None:
        super(Encoder, self).__init__()
        layers = [
            EncoderBlock(C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         padding=P[i],
                         cplx=cplx,
                         causal=causal) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)

    def forward(self, x: th.Tensor) -> Tuple[List[th.Tensor], th.Tensor]:
        enc_h = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
            # print(f"encoder-{index}: {x.shape}")
            if index + 1 != self.num_layers:
                enc_h.append(x)
        return enc_h, x


class Decoder(nn.Module):
    """
    Decoder of the UNet
        K: filters
        S: strides
        C: output channels
    """

    def __init__(self,
                 cplx: bool,
                 K: List[Tuple[int, int]],
                 S: List[Tuple[int, int]],
                 C: List[int],
                 P: List[int],
                 O: List[int],
                 causal: bool = False,
                 connection: str = "sum") -> None:
        super(Decoder, self).__init__()
        if connection not in ["cat", "sum"]:
            raise ValueError(f"Unknown connection mode: {connection}")
        layers = [
            DecoderBlock(C[i] * 2 if connection == "cat" and i != 0 else C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         padding=P[i],
                         output_padding=O[i],
                         causal=causal,
                         cplx=cplx,
                         last_layer=(i == len(K) - 1)) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)
        self.connection = connection

    def forward(self, x: th.Tensor, enc_h: List[th.Tensor]) -> th.Tensor:
        # N = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(x)
            else:
                # N x C x F x T
                if self.connection == "sum":
                    inp = x + enc_h[index - 1]
                else:
                    # N x 2C x F x T
                    inp = th.cat([x, enc_h[index - 1]], 1)
                x = layer(inp)
            # print(f"decoder-{N - 1 - index}: {x.shape}")
        return x


@ApsRegisters.sse.register("sse@dcunet")
class DCUNet(SseBase):
    """
    Real or Complex UNet for Speech Enhancement

    Args:
        K, S, C: kernel, stride, padding, channel size for convolution in encoder/decoder
        P: padding on frequency axis for convolution in encoder/decoder
        O: output_padding on frequency axis for transposed_conv2d in decoder
    NOTE: make sure that stride size on time axis is 1 (we do not do subsampling on time axis)
    """

    def __init__(self,
                 cplx: bool = True,
                 K: str = "7,5;7,5;7,5;5,3;5,3;5,3;5,3",
                 S: str = "2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 C: str = "32,32,64,64,64,64,64",
                 P: str = "1,1,1,1,1,1,1",
                 O: str = "0,0,0,0,0,0,0",
                 num_branch: int = 1,
                 causal_conv: bool = False,
                 enh_transform: Optional[nn.Module] = None,
                 freq_padding: bool = True,
                 connection: str = "sum") -> None:
        super(DCUNet, self).__init__(enh_transform, training_mode="freq")
        assert enh_transform is not None
        self.cplx = cplx
        self.forward_stft = enh_transform.ctx(name="forward_stft")
        self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        K = parse_2dstr(K)
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        P = parse_1dstr(P)
        O = parse_1dstr(O)
        self.encoder = Encoder(cplx, K, S, [1] + C, P, causal=causal_conv)
        self.decoder = Decoder(cplx,
                               K[::-1],
                               S[::-1],
                               C[::-1] + [num_branch],
                               P[::-1],
                               O[::-1],
                               causal=causal_conv,
                               connection=connection)
        self.num_branch = num_branch

    def sep(self, m: th.Tensor, sr: th.Tensor, si: th.Tensor) -> th.Tensor:
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2)**0.5
            m_mag = th.tanh(m_abs)
            mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
            s = self.inverse_stft((sr * mr - si * mi, sr * mi + si * mr),
                                  input="complex")
        else:
            s = self.inverse_stft((sr * m, si * m), input="complex")
        return s

    def infer(self,
              mix: th.Tensor,
              mode="time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            sep = self.forward(mix)
            if self.num_branch == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def forward(self, s: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S
        """
        self.check_args(s, training=True, valid_dim=[2])
        # N x F x T
        sr, si = self.forward_stft(s, output="complex")
        if self.cplx:
            # N x 2F x T
            s = th.cat([sr, si], -2)
        else:
            # N x F x T
            s = (sr**2 + si**2)**0.5
        # encoder
        enc_h, h = self.encoder(s[:, None])
        # reverse
        enc_h = enc_h[::-1]
        # decoder
        m = self.decoder(h, enc_h)
        # N x C x 2F x T
        if self.num_branch == 1:
            s = self.sep(m[:, 0], sr, si)
        else:
            s = [self.sep(m[:, i], sr, si) for i in range(self.num_branch)]
        return s
