# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional, Union, List
from aps.sse.base import SseBase, MaskNonLinear
from aps.transform.enh import TFTransposeTransform
from aps.libs import ApsRegisters


def signal_mix_consistency(
        mix: th.Tensor, sep: List[th.Tensor],
        weight: Optional[List[th.Tensor]]) -> List[th.Tensor]:
    """
    Apply mixture consistency projection to the resulting separated waveforms, which projects
    them such that they sum up to the original mixture
    Args:
        mix (Tensor): N x S
        sep list(Tensor): [N x S, ...]
    return:
        sep list(Tensor): [N x S, ...]
    """
    delta = mix - sum(sep)
    if weight is None:
        return [s + delta / len(sep) for s in sep]
    else:
        return [s + delta * w for s, w in zip(sep, weight)]


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


def normalize_layer(norm: str, num_channels: int) -> nn.Module:
    """
    Return the normalize layer
    """
    if norm not in ["cLN", "IN", "gLN", "BN"]:
        raise RuntimeError(f"Unsupported normalize layer: {norm}")
    if norm == "cLN":
        return nn.GroupNorm(1, num_channels)
    elif norm == "IN":
        return nn.GroupNorm(num_channels, num_channels)
    elif norm == "BN":
        return nn.BatchNorm1d(num_channels)
    else:
        return GlobalChannelLayerNorm(num_channels)


class ScaleLinear(nn.Conv1d):
    """
    Linear layer with scale parameters
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 scale_param: float = 1.0):
        super(ScaleLinear, self).__init__(in_features,
                                          out_features,
                                          1,
                                          bias=bias)
        self.scale = nn.Parameter(th.tensor(scale_param)) if scale_param else 1

    def forward(self, inp: th.Tensor):
        out = super().forward(inp)
        return out * self.scale


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
                 scale_param: float = 0,
                 causal: bool = False) -> None:
        super(Conv1dBlock, self).__init__()
        self.pad = dilation * (kernel_size - 1)
        self.cau = causal
        # 1x1 conv
        self.conv1 = ScaleLinear(in_channels,
                                 conv_channels,
                                 scale_param=scale_param)
        self.norm1 = nn.Sequential(nn.PReLU(),
                                   normalize_layer(norm, conv_channels))
        self.dconv = nn.Conv1d(conv_channels,
                               conv_channels,
                               kernel_size,
                               groups=conv_channels,
                               padding=self.pad if causal else self.pad // 2,
                               dilation=dilation)
        self.norm2 = nn.Sequential(nn.PReLU(),
                                   normalize_layer(norm, conv_channels))
        self.conv2 = ScaleLinear(conv_channels,
                                 in_channels,
                                 scale_param=scale_param)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T
        Return:
            out (Tensor): N x C x T
        """
        out = self.norm1(self.conv1(inp))
        out = self.dconv(out)
        if self.cau:
            out = out[..., :-self.pad]
        out = self.norm2(out)
        out = self.conv2(out)
        return out + inp


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
                 skip_residual: bool = True,
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
                            dilation=2**n,
                            scale_param=0 if scaling_param else 0.9**n)
                for n in range(blocks_per_repeat)
            ])
            repeats.append(block)
        self.repeat = nn.Sequential(*repeats)
        self.skip_residual = skip_residual
        if skip_residual:
            tot = num_repeats * (num_repeats - 1) // 2
            self.skip_linear = nn.ModuleList([
                ScaleLinear(in_channels, in_channels, scale_param=1.0)
                for _ in range(tot)
            ])
        else:
            self.skip_linear = None

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x T
        Return:
            out (Tensor): N x C x T
        """
        if self.skip_residual:
            outputs = [inp]
            skip_index = 0
            for index, layer in enumerate(self.repeat):
                for i in range(index):
                    inp += self.skip_linear[skip_index](outputs[i])
                    skip_index += 1
                inp = layer(inp)
                outputs.append(inp)
        else:
            inp = self.repeat(inp)
        return inp


@ApsRegisters.sse.register("sse@time_tcn")
class TimeConvTasNet(SseBase):
    """
    Reference:
        1) Y. Luo, N. Mesgarani. Conv-tasnet: Surpassing Ideal Time–frequency Magnitude
        Masking for Speech Separation[J]. IEEE/ACM transactions on audio, speech,
        and language processing, 2019, 27(8):1256–1266.
        2) I. Kavalerov, S. Wisdom, H. Erdogan, B. Patton, K. Wilson, J. Le Roux, and J. R. Hershey.
        Universal sound separation. In Proc. IEEE Workshop on Applications of Signal Processing to
        Audio and Acoustics (WASPAA), 2019
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
                 causal: bool = False,
                 num_spks: int = 2,
                 non_linear: str = "relu",
                 scaling_param: bool = False,
                 skip_residual: bool = False,
                 mixture_consistency: str = "none") -> None:
        super(TimeConvTasNet, self).__init__(None, training_mode="time")
        assert mixture_consistency in ["none", "fix", "mag", "learn"]
        self.non_linear_type = non_linear
        self.non_linear = MaskNonLinear(non_linear,
                                        enable="positive_wo_softplus")
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder = nn.Conv1d(1, N, L, stride=L // 2, padding=0)
        # before repeat blocks, always cLN
        self.ln = normalize_layer("cLN", N)
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
                                 skip_residual=skip_residual,
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
        self.mixture_consistency = mixture_consistency
        if mixture_consistency == "learn":
            self.weight = nn.Linear(num_spks * N, num_spks)

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
            return sep[0] if self.num_spks == 1 else [s[0] for s in sep]

    def mix_consistency(self, out: th.Tensor, mix: th.Tensor,
                        bss: List[th.Tensor]) -> List[th.Tensor]:
        """
        NOTE: current not working
        Args:
            out (Tensor): N x 2F x T
            mix (Tensor): N x S
            sep list(Tensor): [N x S, ...]
        """
        if self.mixture_consistency == "fix":
            weight = None
        elif self.mixture_consistency == "mag":
            mix_sum = th.sum(mix, -1, keepdim=True)
            # [N x 1, ...]
            weight = [th.mean(s**2, -1, keepdim=True) / mix_sum for s in bss]
        else:
            # N x 2F => N x 2
            weight = tf.softmax(self.weight(th.mean(out, -1)), -1)
            # [N x 1, ...]
            weight = th.chunk(weight, self.num_spks, -1)
        # apply signal level mixture consistency
        return signal_mix_consistency(mix, bss, weight)

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
        e = self.mask(y)
        m = th.chunk(e, self.num_spks, 1)
        # 2 x n x N x T
        m = th.stack(m, dim=0)
        # n x N x T
        if self.non_linear_type == "softmax":
            m = self.non_linear(m, dim=0)
        else:
            m = self.non_linear(m,)
        # spks x [n x N x T]
        s = [w * m[n] for n in range(self.num_spks)]
        # spks x n x S
        bss = [self.decoder(x)[:, 0] for x in s]
        if self.mixture_consistency != "none":
            bss = self.mix_consistency(e, mix, bss)
        return bss[0] if self.num_spks == 1 else bss


@ApsRegisters.sse.register("sse@freq_tcn")
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
                 scaling_param: bool = False,
                 skip_residual: bool = False,
                 training_mode: str = "freq") -> None:
        super(FreqConvTasNet, self).__init__(enh_transform,
                                             training_mode=training_mode)
        assert enh_transform is not None
        self.non_linear = MaskNonLinear(non_linear, enable="common")
        self.proj = nn.Sequential(TFTransposeTransform(),
                                  nn.Conv1d(in_features, proj_channels, 1))
        # n x B x T => n x B x T
        self.conv = Conv1dRepeat(N,
                                 B,
                                 in_channels=proj_channels,
                                 conv_channels=conv_channels,
                                 kernel_size=K,
                                 causal=causal,
                                 scaling_param=scaling_param,
                                 skip_residual=skip_residual,
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
        # N x C x T
        x = self.proj(mix_feat)
        # n x B x T
        x = self.conv(x)
        # N x F* x T
        masks = self.non_linear(self.mask(x))
        # [N x F x T, ...]
        masks = th.chunk(masks, self.num_spks, 1)
        if mode == "time":
            decoder = self.enh_transform.inverse_stft
            bss_stft = [mix_stft * m for m in masks]
            bss = [decoder((s.real, s.imag), input="complex") for s in bss_stft]
        else:
            bss = masks
        return bss[0] if self.num_spks == 1 else bss

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
