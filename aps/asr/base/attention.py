#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

import torch.nn.functional as tf

from typing import Optional, Tuple
from aps.libs import Register

AsrAtt = Register("asr_att")


def padding_mask(vec: th.Tensor, device: th.device = None) -> th.Tensor:
    """
    Generate padding masks

    In [1]: a = th.tensor([5, 3, 2, 6, 1])
    In [2]: padding_mask(a)
    Out[2]:
    tensor([[False, False, False, False, False,  True],
            [False, False, False,  True,  True,  True],
            [False, False,  True,  True,  True,  True],
            [False, False, False, False, False, False],
            [False,  True,  True,  True,  True,  True]])
    """
    N = vec.nelement()
    # vector may not in sorted order
    M = vec.max().item()
    templ = th.arange(M, device=vec.device).repeat([N, 1])
    mask = (templ >= vec.unsqueeze(1))
    return mask.to(device) if device is not None else mask


def att_instance(att_type: str, enc_dim: int, dec_dim: int,
                 **kwargs) -> th.nn.Module:
    """
    Return attention instance
    """
    if att_type not in AsrAtt:
        raise RuntimeError(f"Unknown attention type: {att_type}")
    return AsrAtt[att_type](enc_dim, dec_dim, **kwargs)


class Attention(nn.Module):
    """
    Base module for attention
    """

    def __init__(self):
        super(Attention, self).__init__()

    def softmax(self, score: th.Tensor, enc_len: Optional[th.Tensor],
                pad_mask: Optional[th.Tensor]) -> th.Tensor:
        """
        Apply softmax and return alignment
        """
        if enc_len is None:
            return tf.softmax(score, dim=-1)
        else:
            # if enc_len is not None, zero the items that out-of-range
            if pad_mask is None:
                raise RuntimeError("Attention: pad_mask should not be None "
                                   "when enc_len is not None")
            score = score.masked_fill(pad_mask, float("-inf"))
            return tf.softmax(score, dim=-1)

    def clear(self):
        raise NotImplementedError


@AsrAtt.register("loc")
class LocAttention(Attention):
    """
    Location aware attention described in "Attention-Based Models for Speech Recognition"
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int = 512,
                 conv_channels: int = 10,
                 loc_context: int = 64):
        super(LocAttention, self).__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim, bias=False)
        # N x D_conv x T => N x D_att x T
        self.att = nn.Conv1d(conv_channels, att_dim, 1, bias=False)
        # N x 1 x T => N x D_att x T
        self.F = nn.Conv1d(1,
                           conv_channels,
                           loc_context * 2 + 1,
                           stride=1,
                           padding=loc_context)
        self.w = nn.Linear(att_dim, 1, bias=False)
        # clear variables
        self.clear()

    def clear(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
        if self.enc_part is None:
            # N x Ti x D_att
            self.enc_part = self.enc_proj(enc_pad)
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)
        if ali_prev is None:
            # initialize attention
            ali_prev = th.ones(N, T, device=enc_pad.device)
            if enc_len is not None:
                ali_prev = ali_prev.masked_fill(self.pad_mask, 0)
                ali_prev = ali_prev / enc_len[..., None]
            else:
                ali_prev = ali_prev / T
        # N x 1 x T => N x D_conv x Ti
        att_part = self.F(ali_prev[:, None])
        # N x D_conv x Ti => N x D_att x Ti
        att_part = self.att(att_part)
        # N x D_att x Ti => N x Ti x D_att
        att_part = th.transpose(att_part, 1, 2)
        # N x D_dec =>  N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti x D_att
        sum_part = th.tanh(att_part + dec_part[:, None] + self.enc_part)
        # N x Ti
        score = self.w(sum_part).squeeze(-1)
        # ali: N x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # ctx: N x D_enc
        ctx = th.sum(ali[..., None] * enc_pad, 1)
        # return alignment weight & context
        return ali, ctx


@AsrAtt.register("ctx")
class CtxAttention(Attention):
    """
    Context attention described in
        "Neural Machine Translation by Jointly Learning to Align and Translate"
    """

    def __init__(self, enc_dim: int, dec_dim: int, att_dim: int = 512):
        super(CtxAttention, self).__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim, bias=False)
        self.w = nn.Linear(att_dim, 1, bias=False)
        # self.dec_dim = dec_dim
        self.clear()

    def clear(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)
        # N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti x D_att
        sum_part = th.tanh(self.enc_part + dec_part[:, None])
        # N x Ti
        score = self.w(sum_part).squeeze(-1)
        # ali: N x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # ctx: N x D_enc
        ctx = th.sum(ali[..., None] * enc_pad, 1)
        # return alignment weight & context
        return ali, ctx


@AsrAtt.register("dot")
class DotAttention(Attention):
    """
    Dot attention described in
        "Listen, Attend and Spell: A Neural Network for Large "
        "Vocabulary Conversational Speech Recognition"
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int = 512,
                 scaled: bool = True):
        super(DotAttention, self).__init__()
        self.enc_proj = nn.Linear(enc_dim, att_dim)
        self.dec_proj = nn.Linear(dec_dim, att_dim)
        self.att_dim = att_dim
        self.scaled = scaled
        self.clear()

    def clear(self):
        self.enc_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Args
            enc_pad: N x Ti x D_enc
            enc_len: N or None
            dec_prev: N x D_dec
            ali_prev: N x Ti (do not use here)
        Return
            ali: N x Ti
            ctx: N x D_enc
        """
        # N x Ti x D_att
        if self.enc_part is None:
            self.enc_part = self.enc_proj(enc_pad)
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)
        # N x D_att
        dec_part = self.dec_proj(dec_prev)
        # N x Ti
        score = th.bmm(self.enc_part, dec_part[..., None]).squeeze(-1)
        if self.scaled:
            score = score / (self.att_dim**0.5)
        # ali: N x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # ctx: N x D_enc
        ctx = th.sum(ali[..., None] * enc_pad, 1)
        # return alignment weight & context
        return ali, ctx


@AsrAtt.register("mhctx")
class MHCtxAttention(Attention):
    """
    Multi-head context attention
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int = 512,
                 att_head: int = 4):
        super(MHCtxAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head, bias=False)
        # project multi-head context
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        # self.w = nn.ModuleList(
        #     [nn.Linear(att_dim, 1, bias=False) for _ in range(att_head)])
        self.w = nn.Conv1d(att_dim * att_head,
                           att_head,
                           1,
                           groups=att_head,
                           bias=False)
        self.att_dim = att_dim
        self.att_head = att_head
        self.clear()

    def clear(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x H*D_att x Ti
            kp = kp.transpose(1, 2)
            # N x H x D_att x Ti
            self.key_part = kp.view(N, self.att_head, self.att_dim, T)
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)[:, None]
        # N x H*D_att, query
        dec_part = self.dec_proj(dec_prev)
        # N x H x D_att
        dec_part = dec_part.view(-1, self.att_head, self.att_dim)
        # N x H x D_att x Ti
        sum_part = th.tanh(self.key_part + dec_part[..., None])
        # N x H x Ti
        score = self.w(sum_part.view(N, -1, T))
        # N x H x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # N x H x D_att
        ctx = th.sum(ali[..., None] * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx


@AsrAtt.register("mhdot")
class MHDotAttention(Attention):
    """
    Multi-head dot attention
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int = 512,
                 att_head: int = 4,
                 scaled: bool = True):
        super(MHDotAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head)
        # project multi-head context
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        self.att_dim = att_dim
        self.att_head = att_head
        self.scaled = scaled
        self.clear()

    def clear(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x Ti x H x D_att
            kp = kp.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.key_part = kp.transpose(1, 2)
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)[:, None]
        # N x H*D_att, query
        dec_part = self.dec_proj(dec_prev)
        # N x H x D_att
        dec_part = dec_part.view(-1, self.att_head, self.att_dim)
        # N x H x Ti x 1
        score = th.matmul(self.key_part, dec_part[..., None]).squeeze(-1)
        # N x H x Ti
        if self.scaled:
            score = score / (self.att_dim**0.5)
        # N x H x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # N x H x D_att
        ctx = th.sum(ali[..., None] * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx


@AsrAtt.register("mhloc")
class MHLocAttention(Attention):
    """
    Multi-head location aware attention
    """

    def __init__(self,
                 enc_dim: int,
                 dec_dim: int,
                 att_dim: int = 512,
                 conv_channels: int = 10,
                 loc_context: int = 64,
                 att_head: int = 4):
        super(MHLocAttention, self).__init__()
        # value, key, query
        self.enc_proj = nn.Linear(enc_dim, att_dim * att_head)
        self.key_proj = nn.Linear(enc_dim, att_dim * att_head, bias=False)
        self.dec_proj = nn.Linear(dec_dim, att_dim * att_head, bias=False)
        # N x D_conv*H x T => N x D_att*H x T
        self.att = nn.Conv1d(conv_channels * att_head,
                             att_dim * att_head,
                             1,
                             groups=att_head,
                             bias=False)
        # N x H x T => N x D_att*H x T
        self.F = nn.Conv1d(att_head,
                           conv_channels * att_head,
                           loc_context * 2 + 1,
                           stride=1,
                           groups=att_head,
                           padding=loc_context)
        self.w = nn.Conv1d(att_dim * att_head,
                           att_head,
                           1,
                           groups=att_head,
                           bias=False)
        self.ctx_proj = nn.Linear(att_dim * att_head, enc_dim)
        self.att_dim = att_dim
        self.att_head = att_head
        # clear variables
        self.clear()

    def clear(self):
        self.enc_part = None
        self.key_part = None
        self.pad_mask = None

    def forward(self, enc_pad: th.Tensor, enc_len: Optional[th.Tensor],
                dec_prev: th.Tensor,
                ali_prev: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
        if self.enc_part is None:
            # N x Ti x H*D_att
            ep = self.enc_proj(enc_pad)
            # N x Ti x H x D_att
            ep = ep.view(N, T, self.att_head, self.att_dim)
            # N x H x Ti x D_att
            self.enc_part = ep.transpose(1, 2)
            # N x Ti x H*D_att
            kp = self.key_proj(enc_pad)
            # N x H*D_att x Ti
            kp = kp.transpose(1, 2)
            # N x H x D_att x Ti
            self.key_part = kp.view(N, self.att_head, self.att_dim, T)
            # init padding mask
            if enc_len is not None:
                self.pad_mask = padding_mask(enc_len, enc_pad.device)[:, None]
        if ali_prev is None:
            # initialize attention
            ali_prev = th.ones(N, self.att_head, T, device=enc_pad.device)
            if enc_len is not None:
                ali_prev = ali_prev.masked_fill(self.pad_mask, 0)
                ali_prev = ali_prev / enc_len[:, None, None]
            else:
                ali_prev = ali_prev / T
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
        sum_part = th.tanh(att_part + self.key_part + dec_part[..., None])
        # N x H x Ti
        score = self.w(sum_part.view(N, -1, T))
        # N x H x Ti
        ali = self.softmax(score, enc_len, self.pad_mask)
        # N x H x D_att
        ctx = th.sum(ali[..., None] * self.enc_part, -2)
        # N x D_enc
        ctx = self.ctx_proj(ctx.view(N, -1))
        return ali, ctx
