#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.sse.enh.dcunet import Encoder, Decoder, parse_1dstr, parse_2dstr


class LSTMP(nn.Module):
    """
    LSTM with real/imag parts
    """

    def __init__(self,
                 in_features,
                 hidden_size,
                 num_layers=2,
                 dropout=0,
                 bidirectional=False,
                 batch_first=True):
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

    def forward(self, inp):
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
                 in_features,
                 hidden_size,
                 num_layers=2,
                 dropout=0,
                 bidirectional=False,
                 batch_first=True):
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

    def forward(self, inp):
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
                 in_features,
                 num_layers=2,
                 dropout=0,
                 hidden_size=512,
                 cplx=True,
                 bidirectional=False):
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

    def forward(self, inp):
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


supported_nonlinear = {"relu": tf.relu, "sigmoid": th.sigmoid, "tanh": th.tanh}


class DCCRN(nn.Module):
    """
    Deep Complex Convolutional-RNN networks
    """

    def __init__(self,
                 cplx=True,
                 K="3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S="2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 P="1,1,1,1,1,1,1",
                 O="0,0,0,0,0,0,0",
                 C="16,32,64,64,128,128,256",
                 num_spks=2,
                 connection="sum",
                 rnn_hidden=512,
                 rnn_layers=2,
                 rnn_resize=1536,
                 rnn_dropout=0,
                 rnn_bidir=False,
                 causal_conv=False,
                 share_decoder=True,
                 enh_transform=None,
                 non_linear="tanh",
                 training_mode="time"):
        """
        Args:
            K, S, C: kernel, stride, padding, channel size for convolution in encoder/decoder
            P: padding on frequency axis for convolution in encoder/decoder
            O: output_padding on frequency axis for transposed_conv2d in decoder
        NOTE: make sure that stride size on time axis is 1 (we do not do subsampling on time axis)
        """
        super(DCCRN, self).__init__()
        if enh_transform is None:
            raise RuntimeError("Missing configuration for enh_transform")
        if non_linear not in supported_nonlinear:
            raise ValueError(f"Unsupported nonlinear: {non_linear}")
        self.cplx = cplx
        self.non_linear = supported_nonlinear[non_linear]
        self.enh_transform = enh_transform
        K = parse_2dstr(K)
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        P = parse_1dstr(P)
        O = parse_1dstr(O)
        self.encoder = Encoder(cplx, K, S, [1] + C, P, causal=causal_conv)
        if connection == "cat":
            C[-1] *= 2
        if share_decoder:
            self.decoder = Decoder(cplx,
                                   K[::-1],
                                   S[::-1],
                                   C[::-1] + [num_spks],
                                   P[::-1],
                                   O[::-1],
                                   causal=causal_conv,
                                   connection=connection)
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
        self.mode = training_mode
        self.share_decoder = share_decoder

    def sep(self, m, sr, si, mode="freq"):
        decoder = self.enh_transform.inverse_stft
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2)**0.5
            m_mag = self.non_linear(m_abs)
            if mode == "freq":
                # s = (si**2 + sr**2)**0.5 * m_mag
                s = m_mag
            else:
                mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
                s = decoder((sr * mr - si * mi, sr * mi + si * mr),
                            input="complex")
        else:
            m = self.non_linear(m)
            if mode == "freq":
                # s = (si**2 + sr**2)**0.5 * m
                s = m
            else:
                s = decoder((sr * m, si * m), input="complex")
        return s

    def check_args(self, mix, training=True):
        if not training and mix.dim() != 1:
            raise RuntimeError("DCCRN expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError("DCCRN expects 2D tensor (training), " +
                               f"got {mix.dim()} instead")

    def infer(self, mix, mode="time"):
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S or F x T
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            sep = self._forward(mix, mode=mode)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def _forward(self, mix, mode="freq"):
        if self.cplx:
            # N x F x T
            sr, si = self.enh_transform.forward_stft(mix, output="complex")
            # N x 2F x T
            s = th.cat([sr, si], -2)
        else:
            # N x F x T
            # s = (sr**2 + si**2)**0.5
            s, stft, _ = self.enh_transform(mix, None)
            # N x F x T
            s = s.transpose(1, 2)
            sr, si = stft.real, stft.imag
        # encoder
        enc_h, h = self.encoder(s[:, None])
        # h: N x C x (2F) x T
        out_h = self.rnn(h)
        if self.connection == "sum":
            h = h + out_h
        else:
            h = th.cat([out_h, h], 1)
        # reverse
        enc_h = enc_h[::-1]
        # decoder
        # N x C x 2F x T
        if self.share_decoder:
            spk_m = self.decoder(h, enc_h)
        else:
            spk_m = th.cat([decoder(h, enc_h) for decoder in self.decoder], 1)
        if self.num_spks == 1:
            return self.sep(spk_m[:, 0], sr, si, mode=mode)
        else:
            return [
                self.sep(spk_m[:, i], sr, si, mode=mode)
                for i in range(self.num_spks)
            ]

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True)
        return self._forward(s, mode=self.mode)