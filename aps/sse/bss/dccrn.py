#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional, Tuple, List, Union
from aps.sse.enh.dcunet import Encoder, Decoder, parse_1dstr, parse_2dstr
from aps.sse.base import SseBase, MaskNonLinear
from aps.const import EPSILON
from aps.libs import ApsRegisters


class LSTMP(nn.Module):
    """
    LSTM with real/imag parts
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 batch_first: bool = True) -> None:
        super(LSTMP, self).__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_size,
                            dropout=dropout,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.proj = nn.Linear(hidden_size * 2 if bidirectional else hidden_size,
                              in_features,
                              bias=False)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x C x F
        Return:
            out (Tensor): N x T x C x F
        """
        N, T, C, _ = inp.shape
        inp = inp.view(N, T, -1)
        out, _ = self.lstm(inp)
        out = self.proj(out)
        return out.view(N, T, C, -1)


class ComplexLSTMP(nn.Module):
    """
    LSTMP real/imag parts
    """

    def __init__(self,
                 in_features: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0,
                 bidirectional: bool = False,
                 batch_first: bool = True) -> None:
        super(ComplexLSTMP, self).__init__()
        self.real = LSTMP(in_features,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        self.imag = LSTMP(in_features,
                          hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=bidirectional,
                          batch_first=batch_first)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x T x C x 2F
        Return:
            out (Tensor): N x T x C x 2F
        """
        # N x T x C x F
        inp_r, inp_i = th.chunk(inp, 2, -1)
        # (a + bi) (c + di) = (ac - bd) + (bc + ad)i
        out_r = self.real(inp_r) - self.imag(inp_i)
        out_i = self.real(inp_i) + self.imag(inp_r)
        # N x T x C x 2F
        out = th.cat([out_r, out_i], -1)
        return out


class LSTMWrapper(nn.Module):
    """
    R/C LSTM
    """

    def __init__(self,
                 in_features: int,
                 num_layers: int = 2,
                 dropout: float = 0,
                 hidden_size: int = 512,
                 cplx: bool = True,
                 bidirectional: bool = False) -> None:
        super(LSTMWrapper, self).__init__()
        if cplx:
            self.lstm = ComplexLSTMP(in_features,
                                     hidden_size,
                                     dropout=dropout,
                                     num_layers=num_layers,
                                     bidirectional=bidirectional,
                                     batch_first=True)
        else:
            self.lstm = LSTMP(in_features,
                              hidden_size,
                              dropout=dropout,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True)

    def forward(self, inp: th.Tensor) -> th.Tensor:
        """
        Args:
            inp (Tensor): N x C x (2)F x T
        Return:
            out (Tensor): N x C x (2)F x T
        """
        # N x T x C x (2)F
        inp = th.einsum("ncft->ntcf", inp)
        # N x T x C x (2)F
        out = self.lstm(inp)
        # N x C x (2)F x T
        return th.einsum("ntcf->ncft", out)


@ApsRegisters.sse.register("sse@dccrn")
class DCCRN(SseBase):
    """
    Deep Complex Convolutional-RNN networks

    Args:
        K, S, C: kernel, stride, padding, channel size for convolution in encoder/decoder
        P: padding on frequency axis for convolution in encoder/decoder
        O: output_padding on frequency axis for transposed_conv2d in decoder
    NOTE: make sure that stride size on time axis is 1 (we do not do subsampling on time axis)
    """

    def __init__(self,
                 cplx: bool = True,
                 K: str = "3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S: str = "2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 P: str = "1,1,1,1,1,1,1",
                 O: str = "0,0,0,0,0,0,0",
                 C: str = "16,32,64,64,128,128,256",
                 num_spks: int = 2,
                 connection: str = "sum",
                 rnn_hidden: int = 512,
                 rnn_layers: int = 2,
                 rnn_resize: int = 1536,
                 rnn_dropout: float = 0,
                 rnn_bidir: bool = False,
                 causal_conv: bool = False,
                 share_decoder: bool = True,
                 enh_transform: Optional[nn.Module] = None,
                 non_linear: str = "tanh",
                 training_mode: str = "time") -> None:
        super(DCCRN, self).__init__(enh_transform, training_mode=training_mode)
        assert enh_transform is not None
        self.cplx = cplx
        self.non_linear = MaskNonLinear(non_linear, enable="all_wo_softmax")
        self.forward_stft = enh_transform.ctx(name="forward_stft")
        self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        K = parse_2dstr(K)
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        P = parse_1dstr(P)
        O = parse_1dstr(O)
        self.encoder = Encoder(cplx, K, S, [1] + C, P, causal=causal_conv)
        if connection == "cat":
            C[-1] *= 2
        # decoder: type of ModuleList for th.jit.export
        if share_decoder:
            self.decoder = nn.ModuleList([
                Decoder(cplx,
                        K[::-1],
                        S[::-1],
                        C[::-1] + [num_spks],
                        P[::-1],
                        O[::-1],
                        causal=causal_conv,
                        connection=connection)
            ])
        else:
            self.decoder = nn.ModuleList([
                Decoder(cplx,
                        K[::-1],
                        S[::-1],
                        C[::-1] + [1],
                        P[::-1],
                        O[::-1],
                        causal=causal_conv,
                        connection=connection) for _ in range(num_spks)
            ])
        self.rnn = LSTMWrapper(rnn_resize // 2 if cplx else rnn_resize,
                               dropout=rnn_dropout,
                               num_layers=rnn_layers,
                               hidden_size=rnn_hidden,
                               bidirectional=rnn_bidir,
                               cplx=cplx)
        self.num_spks = num_spks
        self.connection = connection
        self.share_decoder = share_decoder

    def _sep(self,
             m: th.Tensor,
             sr: th.Tensor,
             si: th.Tensor,
             mode: str = "freq") -> th.Tensor:
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2 + EPSILON)**0.5
            m_mag = self.non_linear(m_abs)
            mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
            if mode == "freq":
                # N x F x T x 2
                s = th.stack([mr, mi], -1)
            else:
                s = self.inverse_stft((sr * mr - si * mi, sr * mi + si * mr),
                                      input="complex")
        else:
            m = self.non_linear(m)
            if mode == "freq":
                s = m
            else:
                s = self.inverse_stft((sr * m, si * m), input="complex")
        return s

    def _infer(self,
               mix: th.Tensor,
               mode: str = "freq") -> Tuple[th.Tensor, List[th.Tensor]]:
        # N x F x T
        sr, si = self.forward_stft(mix, output="complex")
        # N x C x 2F x T
        masks = self._tf_mask(sr, si)
        if self.num_spks == 1:
            return self._sep(masks[:, 0], sr, si, mode=mode)
        else:
            return [
                self._sep(masks[:, i], sr, si, mode=mode)
                for i in range(self.num_spks)
            ]

    def _tf_mask(self,
                 real: th.Tensor,
                 imag: th.Tensor,
                 eps: float = EPSILON) -> th.Tensor:
        """
        TF mask estimation from given features
        """
        if self.cplx:
            # N x 2F x T
            s = th.cat([real, imag], -2)
        else:
            # N x F x T
            s = (real**2 + imag**2 + eps)**0.5
            # s, stft, _ = self.enh_transform(mix, None)
            # sr, si = stft.real, stft.imag
            # N x F x T
            # s = s.transpose(1, 2)
        # encoder
        enc_h, h = self.encoder(s[:, None])
        # h: N x C x (2F) x T
        out_h = self.rnn(h)
        if self.connection == "sum":
            h = h + out_h
        else:
            h = th.cat([out_h, h], 1)
        # reverse
        # enc_h = enc_h[::-1]
        enc_h = [enc_h[-i] for i in range(1, 1 + len(enc_h))]
        # decoder
        # N x C x 2F x T
        if self.share_decoder:
            masks = self.decoder[0](h, enc_h)
        else:
            masks = th.cat([decoder(h, enc_h) for decoder in self.decoder], 1)
        return masks

    def infer(self,
              mix: th.Tensor,
              mode: str = "time") -> Tuple[th.Tensor, List[th.Tensor]]:
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False, valid_dim=[1])
        with th.no_grad():
            mix = mix[None, :]
            sep = self._infer(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    @th.jit.ignore
    def forward(self, s: th.Tensor) -> Tuple[th.Tensor, List[th.Tensor]]:
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True, valid_dim=[2])
        return self._infer(s, mode=self.training_mode)

    @th.jit.export
    def mask_predict(self, stft: th.Tensor, eps: float = EPSILON) -> th.Tensor:
        """
        Args:
            stft (Tensor): real part of STFT, N x T x F (x2)
        Return:
            masks (Tensor): masks of each speaker, N x T x F (x2)
        """
        # N x F x T x 2
        stft = stft.transpose(1, 2)
        # N x C x (2)F x T
        masks = self._tf_mask(stft[..., 0], stft[..., 1], eps=eps)
        # C x N x T x (2)F
        masks = masks.permute(1, 0, 3, 2)
        if self.cplx:
            # [C x N x T x F, ...]
            real, imag = th.chunk(masks, 2, -1)
            m_abs = (real**2 + imag**2 + eps)**0.5
            m_mag = self.non_linear(m_abs)
            real, imag = m_mag * real / m_abs, m_mag * imag / m_abs
            # C x N x T x F x (2)
            masks = th.stack([real, imag], -1)
        else:
            masks = self.non_linear(masks)
        return masks[0] if self.num_spks == 1 else masks
