# wujian@2020
"""
Convolution based multi-channel front-end processing
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from torch_complex.tensor import ComplexTensor


class ComplexConv(nn.Module):
    """
    Complex convolution layer
    """
    def __init__(self, conv_ins, *args, **kwargs):
        super(ComplexConv, self).__init__()
        self.real = conv_ins(*args, **kwargs)
        self.imag = conv_ins(*args, **kwargs)

    def forward(self, x, add_abs=False, eps=1e-5):
        # x: complex tensor
        xr, xi = x.real, x.imag
        br = self.real(xr) - self.imag(xi)
        bi = self.real(xi) + self.imag(xr)
        if not add_abs:
            return ComplexTensor(br, bi)
        else:
            return (br**2 + bi**2 + eps)**0.5


class ComplexConv1d(ComplexConv):
    """
    Complex 1D convolution layer
    """
    def __init__(self, *args, **kwargs):
        super(ComplexConv1d, self).__init__(nn.Conv1d, *args, **kwargs)


class ComplexConv2d(ComplexConv):
    """
    Complex 2D convolution layer
    """
    def __init__(self, *args, **kwargs):
        super(ComplexConv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class TimeInvariantFE(nn.Module):
    """
    Time invariant convolutional front-end (eq beamformer)
    """
    def __init__(self,
                 num_bins=257,
                 num_channels=4,
                 spatial_filters=8,
                 spectra_filters=80,
                 batchnorm=True):
        super(TimeInvariantFE, self).__init__()
        self.conv = ComplexConv1d(num_bins,
                                  num_bins * spatial_filters,
                                  num_channels,
                                  groups=num_bins,
                                  padding=0,
                                  bias=False)
        self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.B = spatial_filters
        self.C = num_channels

    def forward(self, x, eps=1e-5):
        """
        args:
            x: N x C x F x T, complex tensor
        return:
            y: N x T x ..., enhanced features
        """
        N, C, F, T = x.shape
        if C != self.C:
            raise RuntimeError(f"Expect input channel {self.C}, but {C}")
        # N x C x F x T => N x T x F x C
        x = x.transpose(1, 3)
        x = x.contiguous()
        # NT x F x C
        x = x.view(-1, F, C)
        # NT x FB x 1
        b = self.conv(x, add_abs=True, eps=eps)
        # NT x F x B
        b = b.view(-1, F, self.B)
        # NT x B x F
        b = b.transpose(1, 2)
        # NT x B x D
        f = self.proj(b)
        # NT x B x D
        f = th.log(tf.relu(f) + eps)
        # N x T x B x D
        f = f.view(N, T, self.B, -1)
        if self.norm:
            # N x B x T x D
            f = f.transpose(1, 2)
            f = self.norm(f)
            f = f.transpose(1, 2)
        # N x T x BD
        f = f.contiguous()
        f = f.view(N, T, -1)
        return f


class TimeVariantFE(nn.Module):
    """
    Time variant convolutional front-end
    """
    def __init__(self,
                 num_bins=257,
                 num_channels=4,
                 time_reception=11,
                 spatial_filters=8,
                 spectra_filters=80,
                 batchnorm=True):
        super(TimeVariantFE, self).__init__()
        self.conv = ComplexConv2d(num_bins,
                                  num_bins * spatial_filters,
                                  (time_reception, num_channels),
                                  groups=num_bins,
                                  padding=((time_reception - 1) // 2, 0),
                                  bias=False)
        self.proj = nn.Linear(num_bins, spectra_filters, bias=False)
        self.norm = nn.BatchNorm2d(spatial_filters) if batchnorm else None
        self.B = spatial_filters
        self.C = num_channels

    def forward(self, x, eps=1e-5):
        """
        args:
            x: N x C x F x T, complex tensor
        return:
            y: N x T x ..., enhanced features
        """
        N, C, F, T = x.shape
        if C != self.C:
            raise RuntimeError(f"Expect input channel {self.C}, but {C}")
        # N x C x F x T => N x F x T x C
        x = x.permute(0, 2, 3, 1)
        # N x FB x T x 1
        b = self.conv(x, add_abs=True, eps=eps)
        # N x F x B x T
        b = b.view(N, F, self.B, T)
        # N x T x B x F
        b = b.transpose(1, 3)
        # N x T x B x D
        f = self.proj(b)
        # N x T x B x D
        f = th.log(tf.relu(f) + eps)
        if self.norm:
            # N x B x T x D
            f = f.transpose(1, 2)
            f = self.norm(f)
            f = f.transpose(1, 2)
        # N x T x BD
        f = f.contiguous()
        f = f.view(N, T, -1)
        return f


def foo():
    nnet = TimeInvariantFE(num_bins=257)
    N, C, F, T = 10, 4, 257, 100
    r = th.rand(N, C, F, T)
    i = th.rand(N, C, F, T)
    c = ComplexTensor(r, i)
    d = nnet(c)
    print(d.shape)


if __name__ == "__main__":
    foo()