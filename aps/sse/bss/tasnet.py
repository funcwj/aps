# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Union, List, Tuple
from aps.sse.base import SseBase, MaskNonLinear
from aps.libs import ApsRegisters


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError(
                "ChannelWiseLayerNorm accepts 3D tensor as input")
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self,
                 dim: int,
                 eps: float = 1e-05,
                 elementwise_affine: bool = True) -> None:
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError(
                "GlobalChannelLayerNorm accepts 3D tensor as input")
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self) -> str:
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm: str, dim: int) -> nn.Module:
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError(f"Unsupported normalize layer: {norm}")
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1dBlock(nn.Module):
    """
    1D convolutional block in TasNet
    """

    def __init__(self,
                 in_channels: int = 256,
                 conv_channels: int = 512,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 norm: str = "cLN",
                 scalar: float = 0,
                 skip_connection: bool = True,
                 causal: bool = False) -> None:
        super(Conv1dBlock, self).__init__()
        self.pad = dilation * (kernel_size - 1)
        self.cau = causal
        # 1x1 conv
        self.conv1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.norm1 = nn.Sequential(nn.PReLU(), build_norm(norm, conv_channels))
        self.dconv = nn.Conv1d(conv_channels,
                               conv_channels,
                               kernel_size,
                               groups=conv_channels,
                               padding=self.pad if causal else self.pad // 2,
                               dilation=dilation)
        self.norm2 = nn.Sequential(nn.PReLU(), build_norm(norm, conv_channels))
        self.conv2 = nn.Conv1d(conv_channels, in_channels, 1)
        self.conv2_scaler = nn.Parameter(th.tensor(scalar)) if scalar else 1
        if skip_connection:
            self.conv2_skip = nn.Conv1d(conv_channels, in_channels, 1)
        else:
            self.conv2_skip = None

    def forward(self, inp: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor]]:
        """
        Args:
            inp (Tensor): N x C x T
        Return:
            out (Tensor): N x C x T
            out_skip (Tensor): N x C x T or None
        """
        out = self.norm1(self.conv1(inp))
        out = self.dconv(out)
        if self.cau:
            out = out[..., :-self.pad]
        out = self.norm2(out)
        out_skip = self.conv2_skip(out) if self.conv2_skip else None
        out = self.conv2(out) * self.conv2_scaler
        return out + inp, out_skip


class Conv1dRepeat(nn.Module):
    """
    Stack of Conv1d blocks
    """

    def __init__(self,
                 num_repeats: int,
                 blocks_per_repeat: int,
                 in_channels: int = 128,
                 conv_channels: int = 128,
                 kernel_size: int = 3,
                 norm: str = "BN",
                 skip_connection: bool = True,
                 scaling_param: bool = False,
                 causal: bool = False):
        super(Conv1dRepeat, self).__init__()
        repeats = []
        for r in range(num_repeats):
            block = nn.Sequential(*[
                Conv1dBlock(in_channels=in_channels,
                            conv_channels=conv_channels,
                            kernel_size=kernel_size,
                            norm=norm,
                            causal=causal,
                            skip_connection=False if r == num_repeats -
                            1 and n == blocks_per_repeat -
                            1 else skip_connection,
                            dilation=2**n,
                            scalar=0 if scaling_param else 0.9**n)
                for n in range(blocks_per_repeat)
            ])
            repeats.append(block)
        self.repeat = nn.Sequential(*repeats)
        self.skip_connection = skip_connection

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T
        Return:
            out (Tensor): N x C x T
        """
        skips = []
        for block in self.repeat:
            for layer in block:
                inp, out_skip = layer(inp)
                if out_skip is not None:
                    skips.append(out_skip)
        if self.skip_connection:
            return inp + sum(skips)
        else:
            return inp


@ApsRegisters.sse.register("sse@time_tasnet")
class TimeConvTasNet(SseBase):
    """
    Y. Luo, N. Mesgarani. Conv-tasnet: Surpassing Ideal Time–frequency Magnitude
    Masking for Speech Separation[J]. IEEE/ACM transactions on audio, speech,
    and language processing, 2019, 27(8):1256–1266.
    """

    def __init__(self,
                 L: int = 20,
                 N: int = 256,
                 X: int = 8,
                 R: int = 4,
                 B: int = 256,
                 H: int = 512,
                 P: int = 3,
                 norm: str = "BN",
                 num_spks: int = 2,
                 non_linear: str = "relu",
                 scaling_param: bool = False,
                 skip_connection: bool = False,
                 causal: bool = False) -> None:
        super(TimeConvTasNet, self).__init__(None, training_mode="time")
        self.non_linear_type = non_linear
        self.non_linear = MaskNonLinear(non_linear,
                                        enable="positive_wo_softplus")
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder = nn.Conv1d(1, N, L, stride=L // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = build_norm("cLN", N)
        # n x N x T => n x B x T
        self.proj = nn.Conv1d(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.conv = Conv1dRepeat(R,
                                 X,
                                 in_channels=B,
                                 conv_channels=H,
                                 kernel_size=P,
                                 norm=norm,
                                 skip_connection=skip_connection,
                                 scaling_param=scaling_param,
                                 causal=causal)
        # n x B x T => n x 2N x T
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(B, num_spks * N, 1))
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder = nn.ConvTranspose1d(N,
                                          1,
                                          kernel_size=L,
                                          stride=L // 2,
                                          bias=True)
        self.num_spks = num_spks

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            sep ([Tensor, ...]): S
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            # when inference, only one utt
            mix = mix[None, ...]
            sep = self.forward(mix)
            return sep

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True, valid_dim=[2])
        # n x 1 x S => n x N x T
        w = th.relu(self.encoder(mix[:, None]))
        # n x B x T
        y = self.proj(self.ln(w))
        # n x B x T
        y = self.conv(y)
        # n x 2N x T
        e = th.chunk(self.mask(y), self.num_spks, 1)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(th.stack(e, dim=0), dim=0)
        else:
            m = self.non_linear(th.stack(e, dim=0))
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        spk = [self.decoder(x)[:, 0] for x in s]
        return spk[0] if self.num_spks == 1 else spk


@ApsRegisters.sse.register("sse@freq_tasnet")
class FreqConvTasNet(SseBase):
    """
    Frequency domain ConvTasNet
    """

    def __init__(self,
                 enh_transform: Optional[nn.Module] = None,
                 in_features: int = 257,
                 B: int = 6,
                 K: int = 3,
                 N: int = 3,
                 conv_channels: int = 512,
                 proj_channels: int = 256,
                 norm: str = "BN",
                 num_spks: int = 2,
                 num_bins: int = 257,
                 non_linear: str = "relu",
                 causal: bool = False,
                 skip_connection: bool = False,
                 training_mode: str = "freq") -> None:
        super(FreqConvTasNet, self).__init__(enh_transform,
                                             training_mode=training_mode)
        assert enh_transform is not None
        self.enh_transform = enh_transform
        self.non_linear = MaskNonLinear(non_linear, enable="common")
        self.proj = nn.Conv1d(in_features, proj_channels, 1)
        # n x B x T => n x B x T
        self.conv = Conv1dRepeat(N,
                                 B,
                                 in_channels=proj_channels,
                                 conv_channels=conv_channels,
                                 kernel_size=K,
                                 causal=causal,
                                 skip_connection=skip_connection,
                                 scaling_param=scaling_param,
                                 norm=norm)
        self.mask = nn.Sequential(
            nn.PReLU(), nn.Conv1d(proj_channels, num_bins * num_spks, 1))
        self.num_spks = num_spks

    def _forward(self, mix: th.Tensor,
                 mode: str) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Forward function in time|freq mode
        """
        # mix_feat: N x T x F
        # mix_stft: N x (C) x F x T
        mix_feat, mix_stft, _ = self.enh_transform(mix, None)
        # N x F x T
        if mix_stft.dim() == 4:
            mix_stft = mix_stft[:, 0]
        # N x F x T
        mix_feat = th.transpose(mix_feat, 1, 2)
        # N x C x T
        x = self.proj(mix_feat)
        # n x B x T
        x = self.conv(x)
        # N x F* x T
        masks = self.non_linear(self.mask(x))
        if self.num_spks > 1:
            masks = th.chunk(masks, self.num_spks, 1)
        # N x F x T, ...
        if mode == "freq":
            return masks
        else:
            decoder = self.enh_transform.inverse_stft
            if self.num_spks == 1:
                enh_stft = mix_stft * masks
                enh = decoder((enh_stft.real, enh_stft.imag), input="complex")
            else:
                enh_stft = [mix_stft * m for m in masks]
                enh = [
                    decoder((s.real, s.imag), input="complex") for s in enh_stft
                ]
            return enh

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=False, valid_dim=[1, 2])
        with th.no_grad():
            mix = mix[None, :]
            ret = self._forward(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    def forward(self, mix: th.Tensor) -> Union[th.Tensor, List[th.Tensor]]:
        """
        Args
            mix (Tensor): N x (C) x S
        Return
            m (List(Tensor)): [N x F x T, ...] or
            s (List(Tensor)): [N x S, ...]
        """
        self.check_args(mix, training=True, valid_dim=[2, 3])
        return self._forward(mix, mode=self.training_mode)
