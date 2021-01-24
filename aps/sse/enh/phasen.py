#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Union, Tuple
from aps.const import EPSILON
from aps.sse.base import SseBase
from aps.libs import ApsRegisters

batchnorm_non_linear = {
    "relu": tf.relu,
    "sigmoid": th.sigmoid,
    "": lambda inp: inp
}


class PhasenConv2d(nn.Conv2d):
    """
    Conv2d for Phasen (keeping time/frequency dimention not changed)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int]) -> None:
        padding_size = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        super(PhasenConv2d, self).__init__(in_channels,
                                           out_channels,
                                           kernel_size,
                                           stride=(1, 1),
                                           padding=padding_size)


class PhasenBatchNorm1d(nn.BatchNorm1d):
    """
    BatchNorm1d for Phasen (following a non-linear layer)
    """

    def __init__(self, num_features: int, non_linear: str = "relu") -> None:
        super(PhasenBatchNorm1d, self).__init__(num_features)
        self.non_linear = batchnorm_non_linear[non_linear]

    def forward(self, inp: th.Tensor) -> th.Tensor:
        out = super().forward(inp)
        return self.non_linear(out)


class PhasenBatchNorm2d(nn.BatchNorm2d):
    """
    BatchNorm2d for Phasen (following a non-linear layer)
    """

    def __init__(self, num_features: int, non_linear: str = "relu") -> None:
        super(PhasenBatchNorm2d, self).__init__(num_features)
        self.non_linear = batchnorm_non_linear[non_linear]

    def forward(self, inp: th.Tensor) -> th.Tensor:
        out = super().forward(inp)
        return self.non_linear(out)


class PhasenGlobalNorm(nn.Module):
    """
    Global Normalization for Phasen
    """

    def __init__(self,
                 dim: int,
                 eps: float = 1e-05,
                 elementwise_affine: bool = True) -> None:
        super(PhasenGlobalNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(1, dim))
            self.gamma = nn.Parameter(th.ones(1, dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp: N x C x F x T
        Return:
            out: N x C x F x T
        """
        if inp.dim() != 4:
            raise RuntimeError("PhasenGlobalNorm expects 4D tensor as input")
        # N x 1 x 1
        mean = th.mean(inp, (1, 2, 3), keepdim=True)
        var = th.mean((inp - mean)**2, (1, 2, 3), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            out = self.gamma[..., None, None] * (inp - mean) / th.sqrt(
                var + self.eps) + self.beta[..., None, None]
        else:
            out = (inp - mean) / th.sqrt(var + self.eps)
        return out

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class FTBlock(nn.Module):
    """
    Frequency Transformation Block
    """

    def __init__(self,
                 channel_amp: int,
                 num_bins: int = 257,
                 channel_r: int = 5,
                 conv1d_kernel: int = 9) -> None:
        super(FTBlock, self).__init__()
        self.conv1x1_1 = nn.Sequential(
            nn.Conv2d(channel_amp, channel_r, (1, 1)),
            PhasenBatchNorm2d(channel_r, non_linear="relu"))
        self.linear = nn.Conv1d(num_bins, num_bins, 1, bias=False)
        self.conv1d = nn.Sequential(
            nn.Conv1d(num_bins * channel_r,
                      channel_amp,
                      conv1d_kernel,
                      padding=(conv1d_kernel - 1) // 2),
            nn.BatchNorm1d(channel_amp))
        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(2 * channel_amp, channel_amp, (1, 1)),
            PhasenBatchNorm2d(channel_amp, non_linear="relu"))

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp: N x Ca x F x T
        Return:
            out: N x Ca x F x T
        """
        # N x Cr x F x T
        out = self.conv1x1_1(inp)
        N, _, F, T = out.shape
        # N x Cr*F x T
        out = out.view(N, -1, T)
        # N x Ca x T
        att = self.conv1d(out)
        # N x Ca x F x T
        out = att[..., None, :] * inp
        # N*Ca x F x T
        out = out.view(-1, F, T)
        # N*Ca x F x T
        out = self.linear(out)
        # N x Ca x F x T
        out = out.view(N, -1, F, T)
        # N x 2*Ca x F x T
        cat = th.cat([out, inp], 1)
        # N x Ca x F x T
        out = self.conv1x1_2(cat)
        return out


class TSBlock(nn.Module):
    """
    Two Stream Block
    """

    def __init__(self,
                 channel_amp: int,
                 channel_pha: int,
                 num_bins: int = 257,
                 channel_r: int = 5,
                 conv1d_kernel: int = 9) -> None:
        super(TSBlock, self).__init__()
        self.ftb1 = FTBlock(channel_amp,
                            num_bins=num_bins,
                            channel_r=channel_r,
                            conv1d_kernel=conv1d_kernel)
        self.ftb2 = FTBlock(channel_amp,
                            num_bins=num_bins,
                            channel_r=channel_r,
                            conv1d_kernel=conv1d_kernel)
        self.stream_a = nn.Sequential(
            PhasenConv2d(channel_amp, channel_amp, (5, 5)),
            PhasenBatchNorm2d(channel_amp),
            PhasenConv2d(channel_amp, channel_amp, (1, 25)),
            PhasenBatchNorm2d(channel_amp),
            PhasenConv2d(channel_amp, channel_amp, (5, 5)),
            PhasenBatchNorm2d(channel_amp))
        self.stream_p = nn.Sequential(
            PhasenConv2d(channel_pha, channel_pha, (5, 3)),
            PhasenBatchNorm2d(channel_pha),
            PhasenConv2d(channel_pha, channel_pha, (1, 25)),
            PhasenBatchNorm2d(channel_pha))
        self.att_a = nn.Conv2d(channel_pha, channel_amp, (1, 1))
        self.att_p = nn.Conv2d(channel_amp, channel_pha, (1, 1))

    def forward(self, amp_and_pha: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args:
            amp (Tensor): N x Ca x F x T
            pha (Tensor): N x Cp x F x T
        Return:
            The same shape as input
        """
        amp, pha = amp_and_pha
        amp = self.ftb1(amp)
        amp = self.stream_a(amp)
        amp = self.ftb2(amp)

        pha = self.stream_p(pha)

        amp = th.tanh(self.att_a(pha)) * amp
        pha = th.tanh(self.att_p(amp)) * pha
        return (amp, pha)


@ApsRegisters.sse.register("sse@phasen")
class Phasen(SseBase):
    """
    PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network
    """

    def __init__(self,
                 channel_amp: int,
                 channel_pha: int,
                 enh_transform: Optional[nn.Module] = None,
                 num_tsbs: int = 3,
                 num_bins: int = 257,
                 channel_r: int = 5,
                 conv1d_kernel: int = 9,
                 lstm_hidden: int = 256,
                 linear_size: int = 512,
                 training_mode: int = "freq") -> None:
        super(Phasen, self).__init__(enh_transform, training_mode=training_mode)
        assert enh_transform is not None
        self.forward_stft = enh_transform.ctx(name="forward_stft")
        self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        self.tsb = nn.Sequential(*[
            TSBlock(channel_amp,
                    channel_pha,
                    num_bins=num_bins,
                    channel_r=channel_r,
                    conv1d_kernel=conv1d_kernel) for _ in range(num_tsbs)
        ])
        self.conv_a = nn.Sequential(
            PhasenConv2d(2, channel_amp, (7, 1)),
            PhasenBatchNorm2d(channel_amp),
            PhasenConv2d(channel_amp, channel_amp, (1, 7)),
            PhasenBatchNorm2d(channel_amp))
        self.conv_p = nn.Sequential(
            PhasenGlobalNorm(2), PhasenConv2d(2, channel_pha, (3, 5)),
            PhasenGlobalNorm(channel_pha),
            PhasenConv2d(channel_pha, channel_pha, (25, 1)))
        self.conv1x1_a = nn.Conv2d(channel_amp, 8, (1, 1))
        self.blstm_a = nn.LSTM(num_bins * 8,
                               lstm_hidden,
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True)
        self.linear_a = nn.Sequential(
            nn.Conv1d(lstm_hidden * 2, linear_size, 1),
            PhasenBatchNorm1d(linear_size),
            nn.Conv1d(linear_size, linear_size, 1),
            PhasenBatchNorm1d(linear_size),
            nn.Conv1d(linear_size, num_bins, 1),
            PhasenBatchNorm1d(num_bins, non_linear="sigmoid"),
        )
        self.conv1x1_p = nn.Conv2d(channel_pha, 2, (1, 1))
        self.training_mode = training_mode

    def _forward(
            self,
            mix: th.Tensor,
            mode: str = "freq"
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        """
        # N x F x T
        sr, si = self.forward_stft(mix, output="complex")
        # N x 2 x F x T
        inp = th.stack([sr, si], 1)
        # N x Ca x F x T
        amp = self.conv_a(inp)
        # N x Cp x F x T
        pha = self.conv_p(inp)

        amp, pha = self.tsb((amp, pha))
        # N x 8 x F x T
        amp = self.conv1x1_a(amp)
        # N x 2 x F x T
        pha = self.conv1x1_p(pha)
        # N x F x T
        mag = th.sqrt(pha[:, 0]**2 + pha[:, 1]**2 + EPSILON)
        pha = pha / mag[:, None]
        # N x CaxF x T
        N, _, _, T = amp.shape
        amp = amp.view(N, -1, T)
        # N x T x CaxF
        amp = amp.transpose(1, 2)
        # N x T x H
        amp, _ = self.blstm_a(amp)
        # N x H x T
        amp = amp.transpose(1, 2)
        # N x F x T
        mask = self.linear_a(amp)
        sr = sr * mask
        si = si * mask
        pr, pi = pha[:, 0], pha[:, 1]
        pack_cplx = (sr * pr - si * pi, sr * pi + si * pr)
        if mode == "freq":
            return pack_cplx
        else:
            return self.inverse_stft(pack_cplx, input="complex")

    def infer(self, mix: th.Tensor, mode="time") -> th.Tensor:
        """
        Args:
            mix (Tensor): S
        Return:
            sep (Tensor): S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, ...]
            enh = self._forward(mix, mode=mode)
            return enh[0] if mode == "time" else (enh[0][0], enh[1][0])

    def forward(self, mix: th.Tensor):
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        return self._forward(mix, mode=self.training_mode)
