# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
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

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
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

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim, momentum=0)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


def build_blocks(N, B, **kwargs):
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

    def forward(self, x, squeeze=False):
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

    def forward(self, x, squeeze=False):
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
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 causal=False,
                 bias=True,
                 norm="BN"):
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

    def forward(self, x):
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
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
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

    def forward(self, x):
        y = self.conv(x)
        y = self.norm(self.prelu(y))
        y = self.dsconv(y)
        x = x + y
        return x


class TimeConvTasNet(nn.Module):
    """
    Y. Luo, N. Mesgarani. Conv-tasnet: Surpassing Ideal Time–frequency Magnitude 
    Masking for Speech Separation[J]. IEEE/ACM transactions on audio, speech, 
    and language processing, 2019, 27(8):1256–1266.
    """

    def __init__(self,
                 L=20,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="BN",
                 num_spks=2,
                 non_linear="relu",
                 input_norm="cLN",
                 block_residual=False,
                 causal=False):
        super(TimeConvTasNet, self).__init__()
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError(f"Unsupported non-linear function: {non_linear}")
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
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

    def check_args(self, mix, training=True):
        """
        Check args training | inference
        """
        if mix.dim() != 1 and not training:
            raise RuntimeError("ConvTasNet expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if mix.dim() != 2 and training:
            raise RuntimeError(f"ConvTasNet expects 2D tensor (training), " +
                               f"but got {mix.dim()}")

    def infer(self, mix):
        """
        Args:
            mix (Tensor): S
        Return:
            sep ([Tensor, ...]): S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            # when inference, only one utt
            mix = mix[None, ...]
            sep = self.forward(mix)
            return sep

    def forward(self, mix):
        """
        Args:
            mix (Tensor): N x S
        Return:
            [Tensor, ...]: N x S
        """
        self.check_args(mix, training=True)
        # n x 1 x S => n x N x T
        w = F.relu(self.encoder(mix))
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


class FreqConvTasNet(nn.Module):
    """
    Frequency domain ConvTasNet
    """

    def __init__(self,
                 enh_transform=None,
                 in_features=257,
                 B=6,
                 K=3,
                 N=3,
                 conv_channels=512,
                 proj_channels=256,
                 norm="BN",
                 num_spks=2,
                 num_bins=257,
                 non_linear="relu",
                 causal=False,
                 block_residual=False,
                 training_mode="freq"):
        super(FreqConvTasNet, self).__init__()
        supported_nonlinear = {"relu": F.relu, "sigmoid": th.sigmoid}
        if non_linear not in supported_nonlinear:
            raise RuntimeError(f"Unsupported non-linear function: {non_linear}")
        if enh_transform is None:
            raise RuntimeError(
                "FreqConvTasNet: missing configuration for enh_transform")
        if training_mode not in ["time", "freq"]:
            raise ValueError(f"Unsupported mode: {training_mode}")
        self.enh_transform = enh_transform
        self.non_linear = supported_nonlinear[non_linear]
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
        self.mode = training_mode
        self.block_residual = block_residual

    def check_args(self, mix, training=True):
        """
        Check args training | inference
        """
        if not training and mix.dim() not in [1, 2]:
            raise RuntimeError(
                "FreqConvTasNet expects 1/2D tensor (inference), " +
                f"got {mix.dim()} instead")

        if training and mix.dim() not in [2, 3]:
            raise RuntimeError(
                f"FreqConvTasNet expects 2/3D tensor (training), " +
                f"got {mix.dim()} instead")

    def _forward(self, mix, mode):
        """
        Forward function in time|freq mode
        """
        # mix_feat: N x T x F
        # mix_stft: N x (C) x F x T
        mix_feat, mix_stft, _ = self.enh_transform(mix, None)
        if mix_stft.dim() == 4:
            # N x F x T
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

    def infer(self, mix, mode="time"):
        """
        Args:
            mix (Tensor): N x S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            ret = self._forward(mix, mode=mode)
            return ret[0] if self.num_spks == 1 else [r[0] for r in ret]

    def forward(self, mix):
        """
        Args
            mix (Tensor): N x (C) x S
        Return
            m (List(Tensor)): [N x F x T, ...] or
            s (List(Tensor)): [N x S, ...]
        """
        self.check_args(mix, training=True)
        return self._forward(mix, mode=self.mode)
