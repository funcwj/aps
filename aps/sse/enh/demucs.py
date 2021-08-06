#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Reference:
https://github.com/facebookresearch/denoiser
"""
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.sse.base import SSEBase
from aps.libs import ApsRegisters
from aps.const import TORCH_VERSION, EPSILON


def sinc(t):
    if TORCH_VERSION >= 1.8:
        return th.sinc(t)
    else:
        return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype),
                        th.sin(t) / t)


def kernel_sampling(zeros=56):
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    kernel = (sinc(t * math.pi) * winodd).view(1, 1, -1)
    return kernel


def workout_train_chunk_length(inp_len: int,
                               resampling_factor: int = 1,
                               num_encoders: int = 5,
                               kernel: int = 8,
                               stride: int = 2) -> int:
    """
    Given inp_len, return the chunk size for training
    """
    out_len = inp_len * resampling_factor
    for _ in range(num_encoders):
        out_len = math.ceil((out_len - kernel) / stride) + 1
    for _ in range(num_encoders):
        out_len = (out_len - 1) * stride + kernel
    return math.ceil(out_len / resampling_factor)


class SamplingBase(nn.Module):
    """
    Only for re-sampling with 2^N
    """

    def __init__(self, factor: int, zeros: int) -> None:
        super(SamplingBase, self).__init__()
        self.zeros = zeros
        self.factor = int(math.log2(factor))
        self.kernel = nn.Parameter(kernel_sampling(zeros), requires_grad=False)

    def filter(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError()

    def forward(self, signal: th.Tensor) -> th.Tensor:
        for _ in range(self.factor):
            signal = self.filter(signal)
        return signal


class Upsampling(SamplingBase):
    """
    Upsampling layer
    """

    def __init__(self, factor: int = 2, zeros: int = 56):
        super(Upsampling, self).__init__(factor, zeros)

    def filter(self, x: th.Tensor) -> th.Tensor:
        *other, time = x.shape
        out = tf.conv1d(x.view(-1, 1, time), self.kernel,
                        padding=self.zeros)[..., 1:].view(*other, time)
        out = th.stack([x, out], dim=-1).view(*other, -1)
        return out


class DnSampling(SamplingBase):
    """
    Downsampling layer
    """

    def __init__(self, factor: int = 2, zeros: int = 56):
        super(DnSampling, self).__init__(factor, zeros)

    def filter(self, x: th.Tensor) -> th.Tensor:
        x_pad = tf.pad(x, (0, 1)) if x.shape[-1] % 2 else x
        xeven = x_pad[..., ::2]
        xodd = x_pad[..., 1::2]
        *other, time = xodd.shape
        out = tf.conv1d(xodd.view(-1, 1, time), self.kernel,
                        padding=self.zeros)[..., :-1].view(*other, time)
        out = (xeven + out).view(*other, -1).mul(0.5)
        return out


class Encoder(nn.Module):
    """
    Encoder layer of the DEMUCS
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int,
                 stride: int,
                 activation: str = "relu") -> None:
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride), nn.ReLU(),
            nn.Conv1d(out_channels,
                      out_channels * (2 if activation == "glu" else 1), 1),
            nn.GLU() if activation == "glu" else nn.ReLU())

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return self.conv(inp)


class Decoder(nn.Module):
    """
    Decoder layer of the DEMUCS
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int,
                 stride: int,
                 activation: str = "relu",
                 last_layer: bool = False):
        super(Decoder, self).__init__()
        decoder = [
            nn.Conv1d(in_channels,
                      in_channels * (2 if activation == "glu" else 1), 1),
            nn.GLU() if activation == "glu" else nn.ReLU(),
            nn.ConvTranspose1d(in_channels, out_channels, kernel, stride)
        ]
        if not last_layer:
            decoder.append(nn.ReLU())
        self.conv = nn.Sequential(*decoder)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return self.conv(inp)


@ApsRegisters.sse.register("sse@demucs")
class DEMUCS(SSEBase):
    """
    Reference:
        Defossez A, Synnaeve G, Adi Y. Real time speech enhancement in the waveform domain[J].
        arXiv preprint arXiv:2006.12847, 2020.
    Args:
        channel: convolution channels
        stride: convolution stride
        kernel: convolution kernel size
        num_layers: number layers of encoder/decoder
        resampling_factor: upsampling factor {0, 2, 4}
        bidirectional: use blstm or not
    """

    def __init__(self,
                 channel: int = 64,
                 stride: int = 2,
                 kernel: int = 8,
                 resampling_factor: int = 1,
                 num_layers: int = 5,
                 rnn_layers: int = 2,
                 growth: float = 2,
                 bidirectional: bool = False,
                 rescale: float = 0.1) -> None:
        super(DEMUCS, self).__init__(None, training_mode="time")
        assert resampling_factor in [1, 2, 4]
        if resampling_factor != 1:
            self.upsampling = Upsampling(resampling_factor)
            self.dnsampling = DnSampling(resampling_factor)
        else:
            self.upsampling = None
            self.dnsampling = None
        self.resampling_factor = resampling_factor
        self.kernel = kernel
        H = channel
        self.encoder = nn.ModuleList([
            Encoder(1 if i == 0 else int(H * growth**(i - 1)),
                    int(H * growth**i), kernel, stride)
            for i in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            Decoder(int(H * growth**i),
                    1 if i == 0 else int(H * growth**(i - 1)),
                    kernel,
                    stride,
                    last_layer=i == 0) for i in range(num_layers - 1, -1, -1)
        ])
        H = int(H * growth**(num_layers - 1))
        self.lstm = nn.LSTM(H,
                            H,
                            num_layers=rnn_layers,
                            bidirectional=bidirectional)
        self.proj = nn.Linear(2 * H, H) if bidirectional else None
        self.reset_parameters(rescale)

    def reset_parameters(self, rescale: float) -> None:
        """
        Reset conv1d/convtranspose1d parameters
        """
        for sub in self.modules():
            if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
                std = sub.weight.std().detach()
                scale = (std / rescale)**0.5
                sub.weight.data /= scale
                sub.bias.data /= scale

    def infer(self, mix: th.Tensor, mode: str = "time") -> th.Tensor:
        """
        Args:
            mix (Tensor): S
        Return:
            enh (Tensor): S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            inp_len = mix.shape[-1]
            pad = workout_train_chunk_length(inp_len) - inp_len
            inp = tf.pad(mix, (0, pad)) if pad else mix
            enh = self.forward(inp[None, ...])
            return enh[0, :inp_len]

    def forward(self, mix: th.Tensor) -> th.Tensor:
        """
        Args:
            mix (Tensor): N x S
        Return:
            enh (Tensor): N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        enc_out = []
        std, _ = th.std_mean(mix, dim=-1, keepdim=True)
        mix = mix / (std + EPSILON)
        # N x S => N x 1 x S
        out = mix.unsqueeze(1)
        if self.upsampling:
            out = self.upsampling(out)
        for encoder in self.encoder:
            out = encoder(out)
            enc_out.append(out)
        # N x C x T => N x T x C
        inp = out.transpose(1, 2)
        inp, _ = self.lstm(inp)
        if self.proj:
            inp = self.proj(inp)
        # N x T x C => N x C x T
        enh = inp.transpose(1, 2)
        # reverse order
        enc_out = enc_out[::-1]
        idx = 0
        for decoder in self.decoder:
            enh = enc_out[idx][..., :enh.shape[-1]] + enh
            enh = decoder(enh)
            idx += 1
        if self.dnsampling:
            enh = self.dnsampling(enh)
        return enh[:, 0] * std
