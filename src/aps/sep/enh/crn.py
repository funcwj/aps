#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as tf


class CRNLayer(nn.Module):
    """
    Encoder/Decoder block of CRN
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride_size=(1, 2),
                 encoder=True,
                 output_layer=False,
                 output_padding=0):
        super(CRNLayer, self).__init__()
        if encoder:
            self.conv2d = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride_size,
                                    padding=(1, 0))
        else:
            self.conv2d = nn.ConvTranspose2d(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride=stride_size,
                                             padding=(1, 0),
                                             output_padding=output_padding)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.output_layer = output_layer

    def forward(self, x):
        """
        Args:
            x (Tensor): N x C x T x F
        """
        x = self.conv2d(x)
        x = self.batchnorm(x)
        if self.output_layer:
            return tf.softplus(x)
        else:
            return tf.elu(x)


class CRNet(nn.Module):
    """
    Reference:
        Tan K. and Wang D.L. (2018): A convolutional recurrent neural network for 
        real-time speech enhancement. Proceedings of INTERSPEECH-18, pp. 3229-3233.
    """
    def __init__(self, num_bins=161, enh_transform=None):
        super(CRNet, self).__init__()
        if num_bins != 161:
            raise RuntimeError(f"Do not support num_bins={num_bins}")
        if not enh_transform:
            raise RuntimeError(
                "Need feature extractor: enh_transform can not be None")
        if enh_transform.feats_dim != num_bins:
            raise RuntimeError(f"Feature dimention != num_bins (num_bins)")
        num_encoders = 5
        K = [16, 32, 64, 128, 256]
        P = [0, (0, 1), 0, 0, 0]
        self.enh_transform = enh_transform
        self.encoders = nn.ModuleList([
            CRNLayer(1 if i == 0 else K[i - 1], K[i], encoder=True)
            for i in range(num_encoders)
        ])
        self.decoders = nn.ModuleList([
            CRNLayer(K[i] * 2,
                     1 if i == 0 else K[i] // 2,
                     encoder=False,
                     output_layer=i == 0,
                     output_padding=P[i])
            for i in range(num_encoders - 1, -1, -1)
        ])
        self.rnns = nn.LSTM(1024,
                            1024,
                            2,
                            batch_first=True,
                            bidirectional=False)

    def infer(self, mix):
        """
        Args:
            mix: (Tensor): N x S
        """
        with th.no_grad():
            if mix.dim() != 1:
                raise RuntimeError("CRNet expects 1D tensor (inference), " +
                                   f"got {mix.dim()} instead")
            mix = mix[None, :]
            # N x T x F
            _, mix_stft, _ = self.enh_transform(mix, None)
            # pha: N x T x F
            pha = mix_stft.angle()
            # mag: N x T x F
            mag = self.forward(mix)
            # enh: N x S
            enh = self.enh_transform.inverse_stft((mag, pha), input="polar")
            return enh[0]

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x S
        """
        if mix.dim() not in [2]:
            raise RuntimeError("CRNet expects 2D tensor (training), " +
                               f"got {mix.dim()} instead")
        # N x T x F
        feats, _, _ = self.enh_transform(mix, None)
        # N x 1 x T x F
        inp = feats[:, None]
        encoder_out = []
        for encoder in self.encoders:
            # N x C x T x F
            inp = encoder(inp)
            encoder_out.append(inp)
        encoder_out = encoder_out[::-1]
        # >>> rnn
        N, C, T, F = inp.shape
        # N x T x C x F
        rnn_inp = inp.transpose(1, 2).contiguous()
        rnn_inp = rnn_inp.view(N, T, -1)
        # N x T x CF
        rnn_out, _ = self.rnns(rnn_inp)
        # N x T x C x F
        rnn_out = rnn_out.view(N, T, C, F)
        # N x C x T x F
        out = rnn_out.transpose(1, 2)
        # <<< rnn
        for i, decoder in enumerate(self.decoders):
            # N x 2C x T x F
            inp = th.cat([out, encoder_out[i]], 1)
            out = decoder(inp)
        # N x T x F => N x F x T
        out = th.transpose(out[:, 0], 1, 2)
        return out