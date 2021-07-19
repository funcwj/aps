#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th
import torch.nn as nn

from typing import Tuple, List, Union, Optional
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters
from aps.const import EPSILON


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


class CasualTruncated(nn.Module):
    """
    Truncated inputs to mimic casual convolutions
    """

    def __init__(self, casual_padding: int) -> None:
        super(CasualTruncated, self).__init__()
        self.padding = casual_padding

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return inp[..., :-self.padding]


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
        # NOTE: time stride should be 1
        time_axis_pad = kernel_size[-1] - 1
        if not causal:
            time_axis_pad = time_axis_pad // 2
        padding = (padding, time_axis_pad)
        ConvClass = ComplexConv2d if cplx else nn.Conv2d
        NormClass = ComplexBatchNorm2d if cplx else nn.BatchNorm2d
        block = [
            ConvClass(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding)
        ]
        if causal:
            block += [CasualTruncated(time_axis_pad)]
        block += [NormClass(out_channels), nn.LeakyReLU()]
        self.block = nn.Sequential(*block)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        return self.block(x)


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
        time_axis_pad = kernel_size[-1] - 1
        if not causal:
            time_axis_pad = time_axis_pad // 2
        padding = (padding, kernel_size[1] - 1 - time_axis_pad)
        ConvClass = ComplexConvTranspose2d if cplx else nn.ConvTranspose2d
        NormClass = ComplexBatchNorm2d if cplx else nn.BatchNorm2d
        block = [
            ConvClass(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding,
                      output_padding=(output_padding, 0))
        ]
        if causal:
            block += [CasualTruncated(time_axis_pad)]
        if not last_layer:
            block += [NormClass(out_channels), nn.LeakyReLU()]
        self.block = nn.Sequential(*block)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x 2C x F x T
        """
        return self.block(x)


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
                 non_linear: str = "tanh",
                 causal_conv: bool = False,
                 enh_transform: Optional[nn.Module] = None,
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
        if not cplx:
            self.non_linear = MaskNonLinear(non_linear, enable="common")
        else:
            self.non_linear = th.tanh
            if non_linear != "tanh":
                warnings.warn(
                    "Given complex=True, we use always use tanh non-linear function"
                )

    def _sep(self, m: th.Tensor, sr: th.Tensor, si: th.Tensor) -> th.Tensor:
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2 + EPSILON)**0.5
            m_mag = self.non_linear(m_abs)
            mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
            s = self.inverse_stft((sr * mr - si * mi, sr * mi + si * mr),
                                  input="complex")
        else:
            m = self.non_linear(m)
            s = self.inverse_stft((sr * m, si * m), input="complex")
        return s

    def _tf_mask(self,
                 real: th.Tensor,
                 imag: th.Tensor,
                 eps: float = EPSILON) -> th.Tensor:
        """
        TF mask estimation from given features
        """
        if self.cplx:
            # N x 2F x T
            inp = th.cat([real, imag], -2)
        else:
            # N x F x T
            inp = (real**2 + imag**2 + eps)**0.5
        # encoder
        enc_h, h = self.encoder(inp[:, None])
        # reverse
        # enc_h = enc_h[::-1]
        enc_h = [enc_h[-i] for i in range(1, 1 + len(enc_h))]
        # decoder
        masks = self.decoder(h, enc_h)
        # N x C x 2F x T
        return masks

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

    @th.jit.ignore
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
        masks = self._tf_mask(sr, si)
        # N x C x 2F x T
        if self.num_branch == 1:
            s = self._sep(masks[:, 0], sr, si)
        else:
            s = [self._sep(masks[:, i], sr, si) for i in range(self.num_branch)]
        return s

    @th.jit.export
    def mask_predict(self, stft: th.Tensor, eps: float = EPSILON) -> th.Tensor:
        """
        Args:
            stft (Tensor): real part of STFT, N x T x F x 2
        Return:
            masks (Tensor): masks of each speaker, C x N x T x F x 2
        """
        # N x F x T x 2
        stft = stft.transpose(1, 2)
        masks = self._tf_mask(stft[..., 0], stft[..., 1], eps=eps)
        # C x N x T x *F
        masks = masks.permute(1, 0, 3, 2)
        if self.cplx:
            # [C x N x T x F, ...]
            real, imag = th.chunk(masks, 2, -1)
            m_abs = (real**2 + imag**2 + eps)**0.5
            m_mag = self.non_linear(m_abs)
            real, imag = m_mag * real / m_abs, m_mag * imag / m_abs
            # C x N x T x F x (2)
            masks = th.stack([real, imag], -1)
        else:
            masks = self.non_linear(masks)
        return masks[0] if self.num_branch == 1 else masks
