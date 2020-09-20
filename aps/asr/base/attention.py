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
        "mhdot": MHDotAttention,
        "mhctx": MHCtxAttention,
        "mhloc": MHLocAttention
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
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            dec_prev: N x D_dec
            ali_prev: N x Ti
        Return
            ali: N x Ti
            ctx: N x D_enc
        """
        N, T, _ = enc_pad.shape
        # prepare variable
        if ali_prev is None:
            # initialize attention
            ali_prev = th.ones(N, T, device=enc_pad.device)
            if enc_len is not None:
                pad_mask = padding_mask(enc_len, enc_pad.device)
                ali_prev = ali_prev.masked_fill(pad_mask, 0)
                ali_prev = ali_prev / enc_len[..., None]
            else:
                ali_prev = ali_prev / T
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
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        Return
            ali: N x Ti
            ctx: N x D_enc
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
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        Return
            ali: N x Ti
            ctx: N x D_enc
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


class MHCtxAttention(nn.Module):
    """
    Multi-head context attention
    """

    def __init__(self, enc_dim, dec_dim, att_dim=512, att_head=4):
        super(MHCtxAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head, bias=False)
        # project multi-head context
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        # self.w = nn.ModuleList(
        #     [nn.Linear(att_dim, 1, bias=False) for _ in range(att_head)])
        self.w = Conv1D(att_dim * att_head,
                        att_head,
                        1,
                        groups=att_head,
                        bias=False)
        self.att_dim = att_dim
        self.att_head = att_head
        self.reset()

    def reset(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        Return
            ali: N x H x Ti
            ctx: N x D_enc
        """
        N, T, _ = enc_pad.shape
        # value
        if self.enc_part is None:
            # N x Ti x H*D_att
            ep = self.enc_proj(enc_pad)
            # N x Ti x H x D_att
            ep = ep.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.enc_part = ep.transpose(1, 2)
        if self.key_part is None:
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x H*D_att x Ti
            kp = kp.transpose(1, 2)
            # N x H x D_att x Ti
            self.key_part = kp.view(N, self.att_head, self.att_dim, T)
        # N x H*D_att, query
        dec_part = self.dec_proj(dec_prev)
        # N x H x D_att
        dec_part = dec_part.view(-1, self.att_head, self.att_dim)
        # N x H x D_att x Ti
        sum_part = th.tanh(self.key_part + dec_part.unsqueeze(-1))
        # N x H x Ti
        e = self.w(sum_part.view(N, -1, T))
        """
        # [N x Ti, ...]
        # e = [vec(sum_part[:, i]).squeeze(-1) for i, vec in enumerate(self.w)]
        # N x H x Ti
        # e = th.stack(e, dim=1)
        """
        # padding
        if enc_len is not None:
            if self.pad_mask is None:
                mask = padding_mask(enc_len, e.device)
                self.pad_mask = mask[:, None]
            e.masked_fill_(self.pad_mask, -float("inf"))
        # softmax N x H x Ti
        ali = F.softmax(e, dim=-1)
        # N x H x D_att
        ctx = th.sum(ali.unsqueeze(-1) * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx


class MHDotAttention(nn.Module):
    """
    Multi-head dot attention
    """

    def __init__(self, enc_dim, dec_dim, att_dim=512, att_head=4):
        super(MHDotAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head)
        # project multi-head context
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        self.att_dim = att_dim
        self.att_head = att_head
        self.reset()

    def reset(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        Return
            ali: N x H x Ti
            ctx: N x D_enc
        """
        N, T, _ = enc_pad.shape
        # value
        if self.enc_part is None:
            # N x Ti x H*D_att
            ep = self.enc_proj(enc_pad)
            # N x Ti x H x D_att
            ep = ep.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.enc_part = ep.transpose(1, 2)
        if self.key_part is None:
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x Ti x H x D_att
            kp = kp.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.key_part = kp.transpose(1, 2)
        # N x H*D_att, query
        dec_part = self.dec_proj(dec_prev)
        # N x H x D_att
        dec_part = dec_part.view(-1, self.att_head, self.att_dim)
        # N x H x Ti x 1
        e = th.matmul(self.key_part, dec_part[..., None])
        # N x H x Ti
        e = e.squeeze(-1) / (self.att_dim**0.5)
        # padding
        if enc_len is not None:
            if self.pad_mask is None:
                mask = padding_mask(enc_len, e.device)
                self.pad_mask = mask[:, None]
            e.masked_fill_(self.pad_mask, -float("inf"))
        # softmax N x H x Ti
        ali = F.softmax(e, dim=-1)
        # N x H x D_att
        ctx = th.sum(ali.unsqueeze(-1) * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx


class MHLocAttention(nn.Module):
    """
    Multi-head location aware attention
    """

    def __init__(self,
                 enc_dim,
                 dec_dim,
                 att_dim=512,
                 att_channels=128,
                 att_kernel=11,
                 att_head=4):
        super(MHLocAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head, bias=False)
        # N x D_conv*H x T => N x D_att*H x T
        self.att = Conv1D(att_channels * att_head,
                          att_dim * att_head,
                          1,
                          groups=att_head,
                          bias=False)
        # N x H x T => N x D_att*H x T
        self.F = Conv1D(att_head,
                        att_channels * att_head,
                        att_kernel,
                        stride=1,
                        groups=att_head,
                        padding=(att_kernel - 1) // 2)
        self.w = Conv1D(att_dim * att_head,
                        att_head,
                        1,
                        groups=att_head,
                        bias=False)
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        self.att_dim = att_dim
        self.att_head = att_head
        # reset variables
        self.reset()

    def reset(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad, enc_len, dec_prev, ali_prev):
        """
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            dec_prev: N x D_dec
            ali_prev: N x H x Ti
        Return
            ali: N x H x Ti
            ctx: N x D_enc
        """
        N, T, _ = enc_pad.shape
        # prepare variable
        if ali_prev is None:
            # initialize attention
            ali_prev = th.ones(N, self.att_head, T, device=enc_pad.device)
            if enc_len is not None:
                pad_mask = padding_mask(enc_len, enc_pad.device)
                ali_prev = ali_prev.masked_fill(pad_mask[:, None], 0)
                ali_prev = ali_prev / enc_len[:, None, None]
            else:
                ali_prev = ali_prev / T
        # value
        if self.enc_part is None:
            # N x Ti x H*D_att
            ep = self.enc_proj(enc_pad)
            # N x Ti x H x D_att
            ep = ep.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.enc_part = ep.transpose(1, 2)
        if self.key_part is None:
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x H*D_att x Ti
            kp = kp.transpose(1, 2)
            # N x H x D_att x Ti
            self.key_part = kp.view(N, self.att_head, self.att_dim, T)

        # N x H x T => N x H*D_conv x Ti
        att_part = self.F(ali_prev)

        # N x H*D_conv x Ti => N x H*D_att x Ti
        att_part = self.att(att_part)
        # N x H x D_att x Ti
        att_part = att_part.view(N, self.att_head, self.att_dim, T)
        # N x D_dec =>  N x H*D_att, query
        dec_part = self.dec_proj(dec_prev)
        # N x H x D_att
        dec_part = dec_part.view(-1, self.att_head, self.att_dim)
        # N x H x D_att x Ti
        sum_part = th.tanh(att_part + self.key_part + dec_part.unsqueeze(-1))
        # N x H x Ti
        e = self.w(sum_part.view(N, -1, T))
        """
        # [N x Ti, ...]
        e = [vec(sum_part[:, i]).squeeze(-1) for i, vec in enumerate(self.w)]
        # N x H x Ti
        e = th.stack(e, dim=1)
        """
        # padding
        if enc_len is not None:
            if self.pad_mask is None:
                mask = padding_mask(enc_len, e.device)
                self.pad_mask = mask[:, None]
            e.masked_fill_(self.pad_mask, -float("inf"))
        # softmax N x H x Ti
        ali = F.softmax(e, dim=-1)
        # N x H x D_att
        ctx = th.sum(ali.unsqueeze(-1) * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx
