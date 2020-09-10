#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F


def parse_1dstr(sstr):
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr):
    return [parse_1dstr(tok) for tok in sstr.split(";")]


class ComplexConv2d(nn.Module):
    """
    Complex 2D Convolution
    """

    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__()
        self.real = nn.Conv2d(*args, **kwargs)
        self.imag = nn.Conv2d(*args, **kwargs)

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
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
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 causal=False,
                 cplx=True):
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

    def forward(self, x):
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
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 output_padding=0,
                 causal=False,
                 cplx=True,
                 last_layer=False):
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

    def forward(self, x):
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

    def __init__(self, cplx, K, S, C, P, causal=False):
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

    def forward(self, x):
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

    def __init__(self, cplx, K, S, C, P, O, causal=False, connection="sum"):
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

    def forward(self, x, enc_h):
        # N = len(self.layers)
        for index, layer in enumerate(self.layers):
            # print(layer)
            if index == 0:
                x = layer(x)
            else:
                if self.connection == "sum":
                    inp = x + enc_h[index - 1]
                else:
                    inp = th.cat([x, enc_h[index - 1]], 1)
                x = layer(inp)
            # print(f"decoder-{N - 1 - index}: {x.shape}")
        return x


class DCUNet(nn.Module):
    """
    Real or Complex UNet for Speech Enhancement
    """

    def __init__(self,
                 cplx=True,
                 K="7,5;7,5;7,5;5,3;5,3;5,3;5,3;5,3",
                 S="2,1;2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 C="32,32,64,64,64,64,64,64",
                 P="1,1,1,1,1,1,1,1",
                 O="0,0,0,0,0,0,0",
                 num_branch=1,
                 causal_conv=False,
                 enh_transform=None,
                 freq_padding=True,
                 connection="sum"):
        super(DCUNet, self).__init__()
        """
        Args:
            K, S, C: kernel, stride, padding, channel size for convolution in encoder/decoder
            P: padding on frequency axis for convolution in encoder/decoder
            O: output_padding on frequency axis for transposed_conv2d in decoder
        NOTE: make sure that stride size on time axis is 1 (we do not do subsampling on time axis)
        """
        if enh_transform is None:
            raise RuntimeError("Missing configuration for enh_transform")
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

    def sep(self, m, sr, si):
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

    def check_args(self, mix, training=True):
        if not training and mix.dim() != 1:
            raise RuntimeError("DCUNet expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError("DCUNet expects 2D tensor (training), " +
                               f"got {mix.dim()} instead")

    def infer(self, mix):
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            sep = self.forward(mix)
            if self.num_branch == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S
        """
        self.check_args(s, training=True)
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
        # print(h.shape)
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


def make_unet(N, cplx=True):
    """
    Return unet with different layers
    """
    if N == 10:
        K = [(7, 5)] * 2 + [(5, 3)] * 3
        S = [(2, 1)] * 5
        C = [1, 32, 64, 64, 64, 64] if cplx else [1, 45, 90, 90, 90, 90]
        # P = [(3, 2)] * 2 + [(2, 1)] * 3
    elif N == 16:
        K = [(7, 5)] * 3 + [(5, 3)] * 5
        S = [(2, 1)] * 8
        C = [1, 32, 32] + [64] * 6 if cplx else [1, 45, 45] + [90] * 6
        # P = [(3, 2)] * 3 + [(2, 1)] * 5
    elif N == 20:
        K = [(7, 1), (1, 7)] + [(7, 5)] * 2 + [(5, 3)] * 6
        S = [(1, 1)] * 2 + [(2, 1)] * 8
        C = [1, 32, 32] + [64] * 7 + [90] if cplx else [1, 45, 45
                                                       ] + [90] * 7 + [180]
        # P = [(3, 0), (0, 3), (3, 2), (3, 2)] + [(2, 1)] * 6
    else:
        raise RuntimeError(f"Unsupported N = {N}")
    return K, S, C
