#!/usr/bin/env python

# wujian@2019

import torch as th
import torch.nn as nn

import torch.nn.functional as F


def padding_mask(vec, device=None):
    """
    Generate padding masks

    In [1]: a = th.tensor([5, 3, 2, 10, 1])
    In [2]: padding_mask(a)
    Out[2]: 
    tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.uint8)
    """
    N = vec.nelement()
    # vector may not in sorted order
    M = vec.max().item()
    templ = th.arange(M, device=vec.device).repeat([N, 1])
    mask = (templ >= vec.unsqueeze(1))
    return mask.to(device) if device is not None else mask


def att_instance(att_type, enc_dim, dec_dim, **kwargs):
    """
    Return attention instance
    """
    supported_att = {
        "dot": DotAttention,
        "loc": LocAttention,
        "ctx": CtxAttention,
        # ...
    }
    if att_type not in supported_att:
        raise RuntimeError("Unknown attention type: {}".format(att_type))
    return supported_att[att_type](enc_dim, dec_dim, **kwargs)


class Conv1D(nn.Conv1d):
    """
    Extend 1D convolution
    """
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class LocAttention(nn.Module):
    """
    Location aware attention described in
        "Attention-Based Models for Speech Recognition"
    """
    def __init__(self,
                 enc_dim,
                 dec_dim,
                 att_dim=512,
                 att_channels=128,
                 att_kernel=11):
        super(LocAttention, self).__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim, bias=False)
        # N x D_conv x T => N x D_att x T
        self.att = Conv1D(att_channels, att_dim, 1, bias=False)
        # N x 1 x T => N x D_att x T
        self.F = Conv1D(1,
                        att_channels,
                        att_kernel,
                        stride=1,
                        padding=(att_kernel - 1) // 2)
        self.w = nn.Linear(att_dim, 1, bias=False)
        # reset variables
        self.reset()

    def reset(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            dec_prev: N x D_dec
            ali_prev: N x Ti
        return
            ali: N x Ti
        """
        N, T, _ = enc_pad.shape
        # prepare variable
        if ali_prev is None:
            # TODO add attention initialization
            ali_prev = th.zeros(N, T, device=enc_pad.device)
        if self.enc_part is None:
            # N x Ti x D_att
            self.enc_part = self.enc_proj(enc_pad)
        # N x T => N x D_conv x Ti
        att_part = self.F(ali_prev)

        # N x D_conv x Ti => N x D_att x Ti
        att_part = self.att(att_part)
        # N x D_att x Ti => N x Ti x D_att
        att_part = th.transpose(att_part, 1, 2)
        # N x D_dec =>  N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti x D_att
        sum_part = th.tanh(att_part + dec_part.unsqueeze(1) + self.enc_part)
        # N x Ti
        e = self.w(sum_part).squeeze(-1)
        # if enc_len is not None, do zero padding
        if enc_len is not None:
            if self.pad_mask is None:
                self.pad_mask = padding_mask(enc_len, e.device)
            e.masked_fill_(self.pad_mask, -float("inf"))
        # attention score/alignment
        # softmax N x Ti
        ali = F.softmax(e, dim=1)
        ctx = th.sum(ali.unsqueeze(-1) * enc_pad, 1)
        # ali: N x Ti
        # ctx: N x D_enc
        return ali, ctx


class CtxAttention(nn.Module):
    """
    Context attention described in
        "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    def __init__(self, enc_dim, dec_dim, att_dim=512):
        super(CtxAttention, self).__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim, bias=False)
        self.w = nn.Linear(att_dim, 1, bias=False)
        # self.dec_dim = dec_dim
        self.reset()

    def reset(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        return
            ali: N x Ti
        """
        # N x Ti x D_att
        if self.enc_part is None:
            self.enc_part = self.enc_proj(enc_pad)
        # N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti x D_att
        sum_part = th.tanh(self.enc_part + dec_part.unsqueeze(1))
        # N x Ti
        e = self.w(sum_part).squeeze(-1)
        # if enc_len is not None, do zero padding
        if enc_len is not None:
            if self.pad_mask is None:
                self.pad_mask = padding_mask(enc_len, e.device)
            e.masked_fill_(self.pad_mask, -float("inf"))
        # attention score/alignment
        # softmax N x Ti
        ali = F.softmax(e, dim=1)
        ctx = th.sum(ali.unsqueeze(-1) * enc_pad, 1)
        # ali: N x Ti
        return ali, ctx


class DotAttention(nn.Module):
    """
    Dot attention described in
        "Listen, Attend and Spell: A Neural Network for Large "
        "Vocabulary Conversational Speech Recognition"
    """
    def __init__(self, enc_dim, dec_dim, att_dim=512):
        super(DotAttention, self).__init__()

        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim)
        self.att_dim = att_dim
        self.reset()

    def reset(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        return
            ali: N x Ti
        """
        # N x Ti x D_att
        if self.enc_part is None:
            self.enc_part = self.enc_proj(enc_pad)
        # N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti
        e = th.bmm(self.enc_part, dec_part.unsqueeze(-1)).squeeze(-1)
        e = e / (self.att_dim**0.5)
        # e = th.bmm(enc_pad, dec_prev.unsqueeze(-1)).squeeze(-1)
        # if enc_len is not None, do zero padding
        if enc_len is not None:
            if self.pad_mask is None:
                self.pad_mask = padding_mask(enc_len, e.device)
            e.masked_fill_(self.pad_mask, -float("inf"))
        # attention score/alignment
        # softmax N x Ti
        ali = F.softmax(e, dim=1)
        ctx = th.sum(ali.unsqueeze(-1) * enc_pad, 1)
        # ali: N x Ti
        return ali, ctx
