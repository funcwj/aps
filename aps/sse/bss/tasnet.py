# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Union, List
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


def build_blocks(N: int, B: int, **kwargs) -> nn.Module:
    """
    Build Conv1D blocks
    """

    def one_block(B, **kwargs):
        blocks = [Conv1DBlock(**kwargs, dilation=(2**n)) for n in range(B)]
        return nn.Sequential(*blocks)

    repeats = [one_block(B, **kwargs) for _ in range(N)]
    return nn.Sequential(*repeats)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x: th.Tensor, squeeze: bool = False) -> th.Tensor:
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Conv1D expects 2/3D tensor as input")
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x: th.Tensor, squeeze: bool = False) -> th.Tensor:
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("ConvTrans1D expects 2/3D tensor as input")
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class DsConv1D(nn.Module):
    """
    Depth-wise separable conv1d block
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 causal: bool = False,
                 bias: bool = True,
                 norm: str = "BN") -> None:
        super(DsConv1D, self).__init__()
        self.dconv_causal = causal
        self.pad_value = dilation * (kernel_size - 1)
        self.dconv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            groups=in_channels,
            padding=self.pad_value if causal else self.pad_value // 2,
            dilation=dilation,
            bias=True)
        self.prelu = nn.PReLU()
        self.norm = build_norm(norm, in_channels)
        self.sconv = nn.Conv1d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.dconv(x)
        if self.dconv_causal:
            x = x[:, :, :-self.pad_value]
        x = self.norm(self.prelu(x))
        x = self.sconv(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block in TasNet
    """

    def __init__(self,
                 in_channels: int = 256,
                 conv_channels: int = 512,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 norm: str = "cLN",
                 causal: bool = False) -> None:
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv = Conv1D(in_channels, conv_channels, 1)
        self.prelu = nn.PReLU()
        self.norm = build_norm(norm, conv_channels)
        self.dsconv = DsConv1D(conv_channels,
                               in_channels,
                               kernel_size,
                               dilation=dilation,
                               causal=causal,
                               bias=True,
                               norm=norm)

    def forward(self, x: th.Tensor) -> th.Tensor:
        y = self.conv(x)
        y = self.norm(self.prelu(y))
        y = self.dsconv(y)
        x = x + y
        return x


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
                 input_norm: str = "cLN",
                 block_residual: bool = False,
                 causal: bool = False) -> None:
        super(TimeConvTasNet, self).__init__(None, training_mode="time")
        self.non_linear_type = non_linear
        self.non_linear = MaskNonLinear(non_linear,
                                        enable="positive_wo_softplus")
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder = Conv1D(1, N, L, stride=L // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = build_norm(input_norm, N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        # repeat blocks
        # n x B x T => n x B x T
        self.conv = build_blocks(R,
                                 X,
                                 in_channels=B,
                                 conv_channels=H,
                                 kernel_size=P,
                                 norm=norm,
                                 causal=causal)
        # n x B x T => n x 2N x T
        self.mask = Conv1D(B, num_spks * N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder = ConvTrans1D(N,
                                   1,
                                   kernel_size=L,
                                   stride=L // 2,
                                   bias=True)
        self.num_spks = num_spks
        self.block_residual = block_residual

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
        w = th.relu(self.encoder(mix))
        # n x B x T
        y = self.proj(self.ln(w))
        # n x B x T
        if self.block_residual:
            for layer in self.conv:
                y = y + layer(y)
        else:
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
        spk = [self.decoder(x, squeeze=True) for x in s]
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
                 block_residual: bool = False,
                 training_mode: str = "freq") -> None:
        super(FreqConvTasNet, self).__init__(enh_transform,
                                             training_mode=training_mode)
        assert enh_transform is not None
        self.enh_transform = enh_transform
        self.non_linear = MaskNonLinear(non_linear, enable="common")
        self.proj = Conv1D(in_features, proj_channels, 1)
        # n x B x T => n x B x T
        self.conv = build_blocks(N,
                                 B,
                                 in_channels=proj_channels,
                                 conv_channels=conv_channels,
                                 kernel_size=K,
                                 causal=causal,
                                 norm=norm)
        self.mask = Conv1D(proj_channels, num_bins * num_spks, 1)
        self.num_spks = num_spks
        self.block_residual = block_residual

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
        if self.block_residual:
            for layer in self.conv:
                x = x + layer(x)
        else:
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
