#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F


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
            x (Tensor): N x C x T x 2F
        Return:
            y (Tensor): N x C' x T' x F'
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
            y (Tensor): N x C' x F' x T'
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
                 cplx=True):
        super(EncoderBlock, self).__init__()
        conv_impl = ComplexConv2d if cplx else nn.Conv2d
        self.conv = conv_impl(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding)
        if cplx:
            self.bn = ComplexBatchNorm2d(out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
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
                 cplx=True,
                 last_layer=False):
        super(DecoderBlock, self).__init__()
        conv_impl = ComplexConvTranspose2d if cplx else nn.ConvTranspose2d
        self.trans_conv = conv_impl(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    output_padding=output_padding)
        if last_layer:
            self.bn = None
        else:
            if cplx:
                self.bn = ComplexBatchNorm2d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.trans_conv(x)
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
    def __init__(self, cplx, K, S, C, P):
        super(Encoder, self).__init__()
        layers = [
            EncoderBlock(C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         cplx=cplx,
                         padding=P[i]) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(layers)

    def forward(self, x):
        enc_h = []
        for index, layer in enumerate(self.layers):
            x = layer(x)
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
    def __init__(self, cplx, K, S, C, P, O):
        super(Decoder, self).__init__()

        layers = [
            DecoderBlock(C[i],
                         C[i + 1],
                         k,
                         stride=S[i],
                         cplx=cplx,
                         padding=P[i],
                         output_padding=O[i],
                         last_layer=(i == len(K) - 1)) for i, k in enumerate(K)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, enc_h):
        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(x)
            else:
                x = layer(x + enc_h[index - 1])
        return x


class DCUNet(nn.Module):
    """
    Real or Complex UNet for Speech Enhancement
    """
    def __init__(self, cplx=True, num_layers=16, enh_transform=None):
        super(DCUNet, self).__init__()
        if enh_transform is None:
            raise RuntimeError("Missing configuration for enh_transform")
        self.cplx = cplx
        self.forward_stft = enh_transform.ctx(name="forward_stft")
        self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        K, S, C, P, O = make_unet(num_layers, cplx=cplx)
        self.encoder = Encoder(cplx, K, S, C, P)
        self.decoder = Decoder(cplx, K[::-1], S[::-1], C[::-1], P[::-1],
                               O[::-1])

    def sep(self, m, sr, si):
        # N x 2F x T
        m = m.squeeze(1)
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

    def infer(self, mix):
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        with th.no_grad():
            if mix.dim() != 1:
                raise RuntimeError("DCUNet expects 1D tensor (inference), " +
                                   f"got {mix.dim()} instead")
            mix = mix[None, :]
            sep = self.forward(mix)
            return sep[0]

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S
        """
        if s.dim() != 2:
            raise RuntimeError("DCUNet expects 2D tensor (training), " +
                               f"got {s.dim()} instead")
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
        s = self.sep(m, sr, si)
        return s


def make_unet(N, cplx=True):
    """
    Return unet with different layers
    """
    if N == 10:
        K = [(7, 5)] * 2 + [(5, 3)] * 3
        S = [(2, 2)] * 4 + [(2, 1)]
        C = [1, 32, 64, 64, 64, 64] if cplx else [1, 45, 90, 90, 90, 90]
        P = [(3, 2)] * 2 + [(2, 1)] * 3
        O = [0, (0, 1), (0, 1), 0, 0]
    elif N == 16:
        K = [(7, 5)] * 3 + [(5, 3)] * 5
        S = [(2, 2), (2, 1)] * 4
        C = [1, 32, 32] + [64] * 6 if cplx else [1, 45, 45] + [90] * 6
        P = [(3, 2)] * 3 + [(2, 1)] * 5
        O = [0, 0, (0, 1), 0, (0, 1), 0, 0, 0]
    elif N == 20:
        K = [(7, 1), (1, 7)] + [(7, 5)] * 2 + [(5, 3)] * 6
        S = [(1, 1)] * 2 + [(2, 2), (2, 1)] * 4
        C = [1, 32, 32] + [64] * 8 + [90] if cplx else [1, 45, 45
                                                        ] + [90] * 8 + [180]
        P = [(3, 0), (0, 3), (3, 2), (3, 2)] + [(2, 1)] * 6
        O = [0, 0, 0, 0, (0, 1), 0, (0, 1), 0, 0, 0]
    else:
        raise RuntimeError(f"Unsupported N = {N}")
    return K, S, C, P, O
