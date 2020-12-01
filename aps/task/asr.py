#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
For ASR task
"""
import math
import torch as th
import torch.nn as nn

import torch.nn.functional as tf

# for RNNT loss, two options:
# https://github.com/HawkAaron/warp-transducer
# https://github.com/1ytic/warp-rnnt
try:
    from warp_rnnt import rnnt_loss as warp_rnnt_objf
    warp_rnnt_available = True
except ImportError:
    warp_rnnt_available = False
try:
    from warprnnt_pytorch import rnnt_loss as warprnnt_pt_objf
    warprnnt_pt_available = True
except ImportError:
    warprnnt_pt_available = False

from typing import Tuple, Dict, NoReturn
from aps.task.base import Task
from aps.task.objf import ce_objf, ls_objf, ctc_objf
from aps.const import IGNORE_ID
from aps.libs import ApsRegisters

__all__ = ["CtcXentHybridTask", "TransducerTask", "LmXentTask"]


def compute_accu(outs: th.Tensor, tgts: th.Tensor) -> float:
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


def prep_asr_target(tgt_pad: th.Tensor,
                    tgt_len: th.Tensor,
                    eos: int = 0) -> Tuple[th.Tensor, th.Tensor]:
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


@ApsRegisters.task.register("ctc_xent")
class CtcXentHybridTask(Task):
    """
    CTC & Attention AM
    """

    def __init__(self,
                 nnet: nn.Module,
                 lsm_factor: float = 0,
                 ctc_weight: float = 0,
                 blank: int = 0) -> None:
        super(CtcXentHybridTask, self).__init__(
            nnet, description="CTC + Xent multi-task training for ASR")
        self.ctc_blank = blank
        self.ctc_weight = ctc_weight
        self.lsm_factor = lsm_factor

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CTC & Attention loss, egs contains:
            src_pad (Tensor): N x Ti x F
            src_len (Tensor): N
            tgt_pad (Tensor): N x To
            tgt_len (Tensor): N
            ssr (float): const if needed
        """
        # tgt_pad: N x To (replace ignore_id with eos)
        # tgts: N x To+1 (add eos)
        tgt_pad, tgts = prep_asr_target(egs["tgt_pad"],
                                        egs["tgt_len"],
                                        eos=self.nnet.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        if "ssr" in egs:
            # with schedule sampling
            outs, _, ctc_enc, enc_len = self.nnet(egs["src_pad"],
                                                  egs["src_len"],
                                                  tgt_pad,
                                                  ssr=egs["ssr"])
        else:
            outs, _, ctc_enc, enc_len = self.nnet(egs["src_pad"],
                                                  egs["src_len"], tgt_pad)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_objf(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_objf(outs, tgts)

        stats = {}
        if self.ctc_weight > 0:
            ctc_loss = ctc_objf(ctc_enc,
                                tgt_pad,
                                enc_len,
                                egs["tgt_len"],
                                blank=self.ctc_blank,
                                add_softmax=True)
            loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * loss
            stats["@ctc"] = ctc_loss.item()
        # compute accu
        accu = compute_accu(outs, tgts)
        # add to reporter
        stats["accu"] = accu
        stats["loss"] = loss
        return stats


@ApsRegisters.task.register("transducer")
class TransducerTask(Task):
    """
    For Transducer based AM
    """

    def __init__(self,
                 nnet: nn.Module,
                 interface: str = "warp_rnnt",
                 blank: int = 0) -> None:
        super(TransducerTask,
              self).__init__(nnet,
                             description="RNNT objective function for ASR")
        self.blank = blank
        self._setup_rnnt_backend(interface)

    def _setup_rnnt_backend(self, interface: str) -> NoReturn:
        """
        Setup RNNT loss impl in the backend
        """
        if interface not in ["warp_rnnt", "warprnnt_pytorch"]:
            raise ValueError(f"Unsupported RNNT interface: {interface}")
        if interface == "warp_rnnt" and not warp_rnnt_available:
            raise ImportError("\"from warp_rnnt import rnnt_loss\" failed")
        if interface == "warprnnt_pytorch" and not warprnnt_pt_available:
            raise ImportError(
                "\"from warprnnt_pytorch import rnnt_loss\" failed")
        self.interface = interface
        self.rnnt_objf = (warp_rnnt_objf
                          if interface == "warp_rnnt" else warprnnt_pt_objf)

    def forward(self, egs: Dict) -> Dict:
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
        rnnt_kwargs = {"blank": self.blank, "reduction": "mean"}
        # add log_softmax if use https://github.com/1ytic/warp-rnnt
        if self.interface == "warp_rnnt":
            outs = tf.log_softmax(outs, -1)
            rnnt_kwargs["gather"] = True
        # compute loss
        loss = self.rnnt_objf(outs, tgt_pad.to(th.int32), enc_len.to(th.int32),
                              egs["tgt_len"].to(th.int32), **rnnt_kwargs)
        return {"loss": loss}


@ApsRegisters.task.register("lm")
class LmXentTask(Task):
    """
    For LM training
    """

    def __init__(self, nnet: nn.Module, repackage_hidden: bool = False) -> None:
        super(LmXentTask, self).__init__(nnet,
                                         description="Xent for LM training")
        self.hidden = None
        self.repackage_hidden = repackage_hidden

    def forward(self, egs: Dict) -> Dict:
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
