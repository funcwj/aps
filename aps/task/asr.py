#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
For ASR task
"""
import math
import numpy as np
import torch as th
import torch.nn as nn

import torch.nn.functional as tf

# https://github.com/HawkAaron/warp-transducer
# https://github.com/1ytic/warp-rnnt
try:
    from warp_rnnt import rnnt_loss
    # from warprnnt_pytorch import rnnt_loss
    rnnt_loss_available = True
except ImportError:
    rnnt_loss_available = False

from aps.task.base import Task
from aps.task.objf import ce_objf, ls_objf

from aps.const import IGNORE_ID

__all__ = ["CtcXentHybridTask", "TransducerTask", "LmXentTask"]


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

    def __init__(self, nnet, lsm_factor=0, ctc_weight=0, blank=0):
        super(CtcXentHybridTask, self).__init__(nnet)
        self.ctc_blank = blank
        self.ctc_weight = ctc_weight
        self.lsm_factor = lsm_factor

    def forward(self, egs, ssr=0, **kwargs):
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
        outs, _, ctc_branch, enc_len = self.nnet(egs["src_pad"],
                                                 egs["src_len"],
                                                 tgt_pad,
                                                 ssr=ssr)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_objf(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_objf(outs, tgts)

        stats = {}
        if self.ctc_weight > 0:
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
            loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * loss
            stats["@ctc"] = ctc_loss.item()
        # compute accu
        accu = compute_accu(outs, tgts)
        # add to reporter
        stats["accu"] = accu
        stats["loss"] = loss
        return stats


class TransducerTask(Task):
    """
    For Transducer based AM
    """

    def __init__(self, nnet, blank=0):
        super(TransducerTask, self).__init__(nnet)
        self.blank = blank
        if not rnnt_loss_available:
            raise ImportError(f"from warp_rnnt import rnnt_loss failed")

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
        outs, enc_len = self.nnet(egs["src_pad"], egs["src_len"], tgt_pad,
                                  egs["tgt_len"])
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
        return {"loss": loss}


class LmXentTask(Task):
    """
    For LM
    """

    def __init__(self, nnet, repackage_hidden=False):
        super(LmXentTask, self).__init__(nnet)
        self.hidden = None
        self.repackage_hidden = repackage_hidden

    def forward(self, egs, **kwargs):
        """
        Compute CE loss, egs contains
            src (Tensor): N x T+1
            tgt (Tensor): N x T+1
            len (Tensor): N
        """
        # pred: N x T+1 x V
        if self.repackage_hidden:
            pred, self.hidden = self.nnet(egs["src"], self.hidden)
        else:
            pred, _ = self.nnet(egs["src"], None, egs["len"])
        loss = ce_objf(pred, egs["tgt"])
        accu = compute_accu(pred, egs["tgt"])
        stats = {"accu": accu, "loss": loss, "@ppl": math.exp(loss.item())}
        return stats
