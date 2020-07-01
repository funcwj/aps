#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn

from ..enh.dcunet import Encoder, Decoder


def parse_1dstr(sstr):
    return list(map(int, sstr.split(",")))


def parse_2dstr(sstr):
    return [parse_1dstr(tok) for tok in sstr.split(";")]


class LSTMP(nn.Module):
    """
    LSTM with real/imag parts
    """
    def __init__(self,
                 in_features,
                 hidden_size,
                 num_layers=2,
                 bidirectional=False,
                 batch_first=True):
        super(LSTMP, self).__init__()
        self.lstm = nn.LSTM(in_features,
                            hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.proj = nn.Linear(hidden_size, in_features, bias=False)

    def forward(self, inp):
        """
        Args:
            inp (Tensor): N x T x F
        Return:
            out (Tensor): N x T x F
        """
        self.lstm.flatten_parameters()
        out, _ = self.lstm(inp)
        return self.proj(out)


class ComplexLSTMP(nn.Module):
    """
    LSTMP real/imag parts
    """
    def __init__(self,
                 in_features,
                 hidden_size,
                 num_layers=2,
                 bidirectional=False,
                 batch_first=True):
        super(ComplexLSTMP, self).__init__()
        self.real = LSTMP(in_features,
                          hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first)
        self.imag = LSTMP(in_features,
                          hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=batch_first)

    def forward(self, inp):
        """
        Args:
            inp (Tensor): N x T x F2
        Return:
            out (Tensor): N x T x F2
        """
        # N x T x F
        inp_r, inp_i = th.chunk(inp, 2, -1)
        # (a + bi) (c + di) = (ac - bd) + (bc + ad)i
        out_r = self.real(inp_r) - self.imag(inp_i)
        out_i = self.real(inp_i) + self.imag(inp_r)
        # N x T x 2F
        out = th.cat([out_r, out_i], -1)
        return out


class LSTMWrapper(nn.Module):
    """
    R/C LSTM
    """
    def __init__(self, in_features, num_layers=2, hidden_size=512, cplx=True):
        super(LSTMWrapper, self).__init__()
        if cplx:
            self.lstm = ComplexLSTMP(in_features,
                                     hidden_size,
                                     num_layers=num_layers,
                                     bidirectional=False,
                                     batch_first=True)
        else:
            self.lstm = LSTMP(in_features,
                              hidden_size,
                              num_layers=num_layers,
                              bidirectional=False,
                              batch_first=True)

    def forward(self, inp):
        """
        Args:
            inp (Tensor): N x C x F(2) x T
        Return:
            out (Tensor): N x C x F(2) x T
        """
        N, C, _, T = inp.shape
        # N x CF2 x T
        inp = inp.view(N, -1, T)
        # N x T x CF2
        inp = inp.transpose(1, -1)
        # N x T x CF2
        out = self.lstm(inp)
        # N x CF2 x T
        out = out.transpose(1, -1)
        # N x C x F2 x T
        return out.view(N, C, -1, T)


class DCRNet(nn.Module):
    """
    Deep Complex CRN
    """
    def __init__(self,
                 cplx=True,
                 K="3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                 S="2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                 C="16,32,64,64,128,128,256",
                 num_spks=2,
                 connection="sum",
                 rnn_hidden=512,
                 rnn_layers=2,
                 rnn_resize=1536,
                 causal_conv=False,
                 enh_transform=None):
        super(DCRNet, self).__init__()
        if enh_transform is None:
            raise RuntimeError("Missing configuration for enh_transform")
        self.cplx = cplx
        self.forward_stft = enh_transform.ctx(name="forward_stft")
        self.inverse_stft = enh_transform.ctx(name="inverse_stft")
        K = parse_2dstr(K)
        # make sure stride size on time axis is 1
        S = parse_2dstr(S)
        C = parse_1dstr(C)
        self.encoder = Encoder(cplx, K, S, [1] + C, causal=causal_conv)
        if connection == "cat":
            C[-1] *= 2
        self.decoder = Decoder(cplx,
                               K[::-1],
                               S[::-1],
                               C[::-1] + [num_spks],
                               causal=causal_conv,
                               connection=connection)
        self.rnn = LSTMWrapper(rnn_resize // 2 if cplx else rnn_resize,
                               num_layers=rnn_layers,
                               hidden_size=rnn_hidden,
                               cplx=cplx)
        self.num_spks = num_spks
        self.connection = connection

    def sep(self, m, sr, si):
        # m: N x 2F x T
        if self.cplx:
            # N x F x T
            mr, mi = th.chunk(m, 2, -2)
            m_abs = (mr**2 + mi**2)**0.5
            m_mag = th.tanh(m_abs)
            mr, mi = m_mag * mr / m_abs, m_mag * mi / m_abs
            s = self.inverse_stft((sr * mr - si * mi, sr * mi + si * mr),
                                  input="complex")
        else:
            s = self.inverse_stft((sr * m, si * m), input="complex")
        return s

    def check_args(self, mix, training=True):
        if not training and mix.dim() != 1:
            raise RuntimeError("DCURN expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError("DCURN expects 2D tensor (training), " +
                               f"got {mix.dim()} instead")

    def infer(self, mix):
        """
        Args:
            mix (Tensor): S
        Return:
            Tensor: S
        """
        self.check_args(mix, training=False)
        with th.no_grad():
            mix = mix[None, :]
            sep = self.forward(mix)
            if self.num_spks == 1:
                return sep[0]
            else:
                return [s[0] for s in sep]

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S
        """
        self.check_args(s, training=True)
        # N x F x T
        sr, si = self.forward_stft(s, output="complex")
        if self.cplx:
            # N x 2F x T
            s = th.cat([sr, si], -2)
        else:
            # N x F x T
            s = (sr**2 + si**2)**0.5
        # encoder
        enc_h, h = self.encoder(s[:, None])
        out_h = self.rnn(h)
        if self.connection == "sum":
            h = h + out_h
        else:
            h = th.cat([out_h, h], 1)
        # reverse
        enc_h = enc_h[::-1]
        # decoder
        # N x C x 2F x T
        spk_m = self.decoder(h, enc_h)
        if self.num_spks == 1:
            return self.sep(spk_m[:, 0], sr, si)
        else:
            return [
                self.sep(spk_m[:, i], sr, si) for i in range(self.num_spks)
            ]
