#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Tuple, Optional
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


class CRNLayer(nn.Module):
    """
    Encoder/Decoder block of CRN
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride_size: Tuple[int, int] = (1, 2),
                 encoder: bool = True,
                 causal: bool = False,
                 output_layer: bool = False,
                 output_padding: int = 0) -> None:
        super(CRNLayer, self).__init__()
        # NOTE: time stride should be 1
        var_kt = kernel_size[0] - 1
        time_axis_pad = var_kt if causal else var_kt // 2
        if encoder:
            self.conv2d = nn.Conv2d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=stride_size,
                                    padding=(time_axis_pad, 0))
        else:
            self.conv2d = nn.ConvTranspose2d(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride=stride_size,
                                             padding=(var_kt - time_axis_pad,
                                                      0),
                                             output_padding=(0, output_padding))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.output_layer = output_layer
        self.causal_conv = causal
        self.time_axis_pad = time_axis_pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (Tensor): N x C x T x F
        """
        x = self.conv2d(x)
        if self.causal_conv:
            x = x[:, :, :-self.time_axis_pad]
        x = self.batchnorm(x)
        if self.output_layer:
            return x
        else:
            return tf.elu(x)


@ApsRegisters.sse.register("sse@crn")
class CRNet(SseBase):
    """
    Reference:
        Tan K. and Wang D.L. (2018): A convolutional recurrent neural network for
        real-time speech enhancement. Proceedings of INTERSPEECH-18, pp. 3229-3233.
    """

    def __init__(self,
                 num_bins: int = 161,
                 causal_conv: bool = False,
                 mode: str = "masking",
                 training_mode: str = "freq",
                 output_nonlinear: str = "softplus",
                 enh_transform: Optional[nn.Module] = None) -> None:
        super(CRNet, self).__init__(enh_transform, training_mode=training_mode)
        assert enh_transform is not None
        if num_bins != 161:
            raise RuntimeError(f"Do not support num_bins={num_bins}")
        if enh_transform.feats_dim != num_bins:
            raise RuntimeError("Feature dimention != num_bins (num_bins)")
        if mode not in ["masking", "mapping"]:
            raise RuntimeError(f"Unsupported mode: {mode}")
        num_encoders = 5
        K = [16, 32, 64, 128, 256]
        P = [0, 1, 0, 0, 0]
        self.out_nonlinear = MaskNonLinear(output_nonlinear,
                                           enable="positive_wo_softmax")
        self.encoders = nn.ModuleList([
            CRNLayer(1 if i == 0 else K[i - 1],
                     K[i],
                     encoder=True,
                     causal=causal_conv) for i in range(num_encoders)
        ])
        self.decoders = nn.ModuleList([
            CRNLayer(K[i] * 2,
                     1 if i == 0 else K[i] // 2,
                     encoder=False,
                     causal=causal_conv,
                     output_layer=i == 0,
                     output_padding=P[i])
            for i in range(num_encoders - 1, -1, -1)
        ])
        self.rnns = nn.LSTM(1024,
                            1024,
                            2,
                            batch_first=True,
                            bidirectional=False)
        self.mode = mode

    def _forward(self, mix: th.Tensor, mode: str = "freq") -> th.Tensor:
        """
        Args:
            mix (Tensor): N x S
        """
        # N x T x F
        feats, mix_stft, _ = self.enh_transform(mix, None)
        # N x 1 x T x F
        inp = feats[:, None]
        encoder_out = []
        for i, encoder in enumerate(self.encoders):
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
        out = self.out_nonlinear(out)
        # N x T x F => N x F x T
        out = th.transpose(out[:, 0], 1, 2)
        if mode == "freq":
            return out
        else:
            decoder = self.enh_transform.inverse_stft
            if self.mode == "masking":
                enh = mix_stft * out
                return decoder((enh.real, enh.imag), input="complex")
            else:
                phase = mix_stft.angle()
                return decoder((out, phase), input="polar")

    def infer(self, mix: th.Tensor, mode: str = "time") -> th.Tensor:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            return self._forward(mix[None, :], mode=mode)[0]

    def forward(self, mix: th.Tensor) -> th.Tensor:
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        return self._forward(mix, mode=self.training_mode)
