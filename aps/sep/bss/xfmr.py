#!/usr/bin/env python

# wujian@2020

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.asr.transformer.encoder import TransformerEncoder, TransformerEncoderLayer
from aps.asr.transformer.encoder import PreNormTransformerEncoderLayer
from aps.asr.transformer.embedding import PositionalEncoding


class TorchTransformer(nn.Module):
    """
    Wrapper for pytorch's Transformer Decoder
    """

    def __init__(self,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 post_norm=True,
                 num_layers=6):
        super(TorchTransformer, self).__init__()
        if post_norm:
            encoder_layer = TransformerEncoderLayer(
                att_dim,
                nhead,
                dim_feedforward=feedforward_dim,
                dropout=att_dropout)
        else:
            encoder_layer = PreNormTransformerEncoderLayer(
                att_dim,
                nhead,
                dim_feedforward=feedforward_dim,
                dropout=att_dropout)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.pe = PositionalEncoding(att_dim, dropout=pos_dropout, max_len=2000)

    def forward(self, inp):
        """
        Args:
            inp: N x T x D
        Return:
            enc_out: N x T x D
        """
        # N x T x F => N x F x T
        inp = self.pe(inp)
        # T x N x D
        enc_out = self.encoder(inp, mask=None, src_key_padding_mask=None)
        # N x T x D
        enc_out = enc_out.transpose(0, 1)
        return enc_out


supported_nonlinear = {"relu": tf.relu, "sigmoid": th.sigmoid}


class FreqTorchXfmr(TorchTransformer):
    """
    Frequency domain Transformer model
    """

    def __init__(self,
                 enh_transform=None,
                 input_size=257,
                 num_spks=2,
                 num_bins=257,
                 att_dim=512,
                 nhead=8,
                 feedforward_dim=2048,
                 pos_dropout=0.1,
                 att_dropout=0.1,
                 proj_dropout=0.1,
                 post_norm=True,
                 num_layers=6,
                 non_linear="sigmoid",
                 training_mode="freq"):
        super(FreqTorchXfmr, self).__init__(att_dim=att_dim,
                                            nhead=nhead,
                                            feedforward_dim=feedforward_dim,
                                            pos_dropout=pos_dropout,
                                            att_dropout=att_dropout,
                                            post_norm=post_norm,
                                            num_layers=num_layers)
        if enh_transform is None:
            raise RuntimeError("enh_transform can not be None")
        if non_linear not in supported_nonlinear:
            raise RuntimeError(f"Unsupported non-linear: {non_linear}")
        self.enh_transform = enh_transform
        self.mode = training_mode
        self.proj = nn.Sequential(nn.Linear(input_size, att_dim),
                                  nn.LayerNorm(att_dim),
                                  nn.Dropout(proj_dropout))
        self.mask = nn.Linear(att_dim, num_bins * num_spks)
        self.non_linear = supported_nonlinear[non_linear]
        self.num_spks = num_spks

    def check_args(self, mix, training=True):
        if not training and mix.dim() != 1:
            raise RuntimeError("FreqTorchXfmr expects 1D tensor (inference), " +
                               f"got {mix.dim()} instead")
        if training and mix.dim() != 2:
            raise RuntimeError("FreqTorchXfmr expects 2D tensor (training), " +
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
        """
        Forward function in time|freq mode
        """
        feats, stft, _ = self.enh_transform(mix, None)
        # stft: N x F x T
        mix = self.proj(feats)
        out = super().forward(mix)
        # N x T x F
        mask = self.non_linear(self.mask(out))
        # N x F x T
        mask = mask.transpose(1, 2)
        if self.num_spks > 1:
            mask = th.chunk(mask, self.num_spks, 1)
        if self.mode == "freq":
            return mask
        else:
            decoder = self.enh_transform.inverse_stft
            if self.num_spks == 1:
                mask = [mask]
            # complex tensor
            spk_stft = [stft * m for m in mask]
            spk = [decoder((s.real, s.imag), input="complex") for s in spk_stft]
            if self.num_spks == 1:
                return spk[0]
            else:
                return spk

    def forward(self, s):
        """
        Args:
            s (Tensor): N x S
        Return:
            Tensor: N x S or N x F x T
        """
        self.check_args(s, training=True)
        return self._forward(s, mode=self.mode)
