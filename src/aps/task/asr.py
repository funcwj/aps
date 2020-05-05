#!/usr/bin/env python

# wujian@2020

import numpy as np
import torch as th
import torch.nn as nn

import torch.nn.functional as tf

# from https://github.com/HawkAaron/warp-transducer
# from warprnnt_pytorch import rnnt_loss
# https://github.com/1ytic/warp-rnnt
from warp_rnnt import rnnt_loss

from .task import Task

IGNORE_ID = -1  # in data loadern

__all__ = ["CtcXentHybridTask", "TransducerTask", "LmXentTask"]


def ce_loss(outs, tgts):
    """
    Cross entropy loss
    """
    _, _, V = outs.shape
    # N(To+1) x V
    outs = outs.view(-1, V)
    # N(To+1)
    tgts = tgts.view(-1)
    ce_loss = tf.cross_entropy(outs,
                               tgts,
                               ignore_index=IGNORE_ID,
                               reduction="mean")
    return ce_loss


def ls_loss(outs, tgts, lsm_factor=0.1):
    """
    Label smooth loss (using KL)
    """
    _, _, V = outs.shape
    # NT x V
    outs = outs.view(-1, V)
    # NT
    tgts = tgts.view(-1)
    mask = (tgts != IGNORE_ID)
    # M x V
    outs = th.masked_select(outs, mask.unsqueeze(-1)).view(-1, V)
    # M
    tgts = th.masked_select(tgts, mask)
    # M x V
    dist = outs.new_full(outs.size(), lsm_factor / V)
    dist = dist.scatter_(1, tgts.unsqueeze(-1), 1 - lsm_factor)
    # KL distance
    loss = tf.kl_div(tf.log_softmax(outs, -1), dist, reduction="batchmean")
    return loss


def compute_accu(outs, tgts):
    """
    Compute frame-level accuracy
    """
    # N x (To+1)
    pred = th.argmax(outs.detach(), dim=-1)
    # ignore mask, -1
    mask = (tgts != IGNORE_ID)
    ncorr = th.sum(pred[mask] == tgts[mask]).float()
    total = th.sum(mask)
    return (ncorr / total).item()


def process_asr_target(tgt_pad, tgt_len, eos=0):
    """
    Process asr targets for inference and loss computation
    """
    # N x To, -1 => EOS
    tgt_v1 = tgt_pad.masked_fill(tgt_pad == IGNORE_ID, eos)
    # N x (To+1), pad -1
    tgt_v2 = tf.pad(tgt_pad, (0, 1), value=IGNORE_ID)
    # add eos
    tgt_v2 = tgt_v2.scatter(1, tgt_len[:, None], eos)
    return tgt_v1, tgt_v2


class CtcXentHybridTask(Task):
    """
    CTC & Attention AM
    """
    def __init__(self, nnet, lsm_factor=0, ctc_regularization=0, ctc_blank=0):
        super(CtcXentHybridTask, self).__init__(nnet)
        self.ctc_blank = ctc_blank
        self.ctc_factor = ctc_regularization
        self.lsm_factor = lsm_factor

    def compute_loss(self, egs, ssr=0, **kwargs):
        """
        Compute CTC & Attention loss, egs contains:
            src_pad (Tensor): N x Ti x F
            src_len (Tensor): N
            tgt_pad (Tensor): N x To
            tgt_len (Tensor): N
        """
        # tgt_pad: N x To (replace ignore_id with eos)
        # tgts: N x To+1 (add eos)
        tgt_pad, tgts = process_asr_target(egs["tgt_pad"],
                                           egs["tgt_len"],
                                           eos=self.nnet.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        pack = (egs["src_pad"], egs["src_len"], tgt_pad, ssr)
        outs, _, ctc_branch, enc_len = self.nnet(pack)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_loss(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_loss(outs, tgts)

        stats = {}
        if self.ctc_factor > 0:
            # add log-softmax, N x T x V => T x N x V
            log_prob = tf.log_softmax(ctc_branch, dim=-1).transpose(0, 1)
            # CTC loss
            ctc_loss = tf.ctc_loss(log_prob,
                                   tgt_pad,
                                   enc_len,
                                   egs["tgt_len"],
                                   blank=self.ctc_blank,
                                   reduction="mean",
                                   zero_infinity=True)
            loss = self.ctc_factor * ctc_loss + (1 - self.ctc_factor) * loss
            stats["fctc"] = ctc_loss.item()
        # compute accu
        accu = compute_accu(outs, tgts)
        # add to reporter
        stats["accu"] = accu
        return loss, stats


class TransducerTask(Task):
    """
    For Transducer based AM
    """
    def __init__(self, nnet, blank=0):
        super(TransducerTask, self).__init__(nnet)
        self.blank = blank

    def forward(self, egs, **kwargs):
        """
        Compute transducer loss, egs contains:
            src_pad (Tensor): N x Ti x F
            src_len (Tensor): N
            tgt_pad (Tensor): N x To
            tgt_len (Tensor): N
        """
        # tgt_pad: N x To (replace ignore_id with blank)
        ignore_mask = egs["tgt_pad"] == IGNORE_ID
        tgt_pad = egs["tgt_pad"].masked_fill(ignore_mask, self.blank)
        # N x Ti x To+1 x V
        pack = (egs["src_pad"], egs["src_len"], tgt_pad, egs["tgt_len"])
        outs, enc_len = self.nnet(pack)
        # add log_softmax if use https://github.com/1ytic/warp-rnnt
        outs = tf.log_softmax(outs, -1)
        # compute loss
        loss = rnnt_loss(outs,
                         tgt_pad.to(th.int32),
                         enc_len.to(th.int32),
                         egs["tgt_len"].to(th.int32),
                         blank=self.blank,
                         reduction="mean",
                         gather=True)
        return loss, None


class LmXentTask(Task):
    """
    For LM
    """
    def __init__(self, nnet, repackage_hidden=False):
        super(LmXentTask, self).__init__(nnet)
        self.hidden = None
        self.repackage_hidden = repackage_hidden

    def compute_loss(self, egs, **kwargs):
        """
        Compute CE loss, egs contains
            src: N x T+1
            tgt: N x T+1
            len: N
        """
        # pred: N x T+1 x V
        if self.repackage_hidden:
            pack = (egs["src"], self.hidden)
            pred, self.hidden = self.nnet(pack)
        else:
            pack = (egs["src"], None, egs["len"])
            pred, _ = self.nnet(pack)
        loss = ce_loss(pred, egs["tgt"])
        accu = compute_accu(pred, egs["tgt"])
        stats = {"accu": accu}
        return loss, stats