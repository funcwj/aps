#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
For ASR task
"""
import warnings
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

from typing import Tuple, Dict, NoReturn, Optional
from aps.task.base import Task
from aps.task.objf import ce_objf, ls_objf, ctc_objf
from aps.const import IGNORE_ID
from aps.libs import ApsRegisters

__all__ = ["CtcXentHybridTask", "TransducerTask", "LmXentTask"]


def compute_accu(outs: th.Tensor, tgts: th.Tensor) -> Tuple[float]:
    """
    Compute frame-level accuracy
    Args:
        outs: N x T, decoder output
        tgts: N x T, padding target labels
    """
    # N x (To+1)
    pred = th.argmax(outs.detach(), dim=-1)
    # ignore mask, -1
    mask = (tgts != IGNORE_ID)
    # numerator
    ncorr = th.sum(pred[mask] == tgts[mask]).float()
    # denumerator
    total = th.sum(mask)
    # return pair
    accu = ncorr / total
    return (accu.item(), total.item())


def prep_asr_label(
        tgt_pad: th.Tensor,
        tgt_len: th.Tensor,
        pad_value: int,
        eos_value: int = -1) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """
    Process asr targets for forward and loss computation
    Args:
        tgt_pad: padding target labels
        tgt_len: target length
        pad_value: padding value, e.g., ignore_id
        eos_value: EOS value
    """
    # N x To, -1 => EOS
    tgt_v1 = tgt_pad.masked_fill(tgt_pad == IGNORE_ID, pad_value)
    # add eos if needed
    if eos_value >= 0:
        # N x (To+1), pad -1
        tgt_v2 = tf.pad(tgt_pad, (0, 1), value=IGNORE_ID)
        # add eos
        tgt_v2 = tgt_v2.scatter(1, tgt_len[:, None], eos_value)
    else:
        tgt_v2 = None
    return tgt_v1, tgt_v2


def load_label_count(label_count: str) -> Optional[th.Tensor]:
    """
    Load tensor from a label count file
    Args:
        label_count: path of the label count file
    """
    if not label_count:
        return None
    counts = []
    with open(label_count, "r") as lc_fd:
        for raw_line in lc_fd:
            toks = raw_line.strip().split()
            num_toks = len(toks)
            if num_toks not in [1, 2]:
                raise RuntimeError(
                    f"Detect format error in label count file: {raw_line}")
            counts.append(float(toks[0] if num_toks == 1 else toks[1]))
    counts = th.tensor(counts)
    num_zeros = th.sum(counts == 0).item()
    if num_zeros:
        warnings.warn(f"Got {num_zeros} labels for zero counting")
    return th.clamp_min(counts, 1)


@ApsRegisters.task.register("ctc_xent")
class CtcXentHybridTask(Task):
    """
    For encoder/decoder attention based AM training. (CTC for encoder, Xent for decoder)
    Args:
        nnet: AM network
        blank: blank id for CTC
        lsm_factor: label smoothing factor
        lsm_method: label smoothing method (uniform|unigram)
        ctc_weight: CTC weight
        label_count: label count file
    """

    def __init__(self,
                 nnet: nn.Module,
                 blank: int = 0,
                 lsm_factor: float = 0,
                 lsm_method: str = "uniform",
                 ctc_weight: float = 0,
                 label_count: str = "") -> None:
        super(CtcXentHybridTask, self).__init__(
            nnet, description="CTC + Xent multi-task training for ASR")
        if lsm_method == "unigram" and not label_count:
            raise RuntimeError(
                "Missing label_count to use unigram label smoothing")
        self.eos = nnet.eos
        self.ctc_blank = blank
        self.ctc_weight = ctc_weight
        self.lsm_factor = lsm_factor
        self.lsm_method = lsm_method
        self.label_count = load_label_count(label_count)

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CTC & Attention loss, egs contains:
            src_pad: N x Ti x F
            src_len: N
            tgt_pad: N x To
            tgt_len: N
            ssr (float): const if needed
        """
        # tgt_pad: N x To (replace ignore_id with eos)
        # tgts: N x To+1 (add eos)
        tgt_pad, tgts = prep_asr_label(egs["tgt_pad"],
                                       egs["tgt_len"],
                                       pad_value=self.eos,
                                       eos_value=self.eos)
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
            xent_loss = ls_objf(outs,
                                tgts,
                                method=self.lsm_method,
                                lsm_factor=self.lsm_factor,
                                label_count=self.label_count)
        else:
            xent_loss = ce_objf(outs, tgts)

        stats = {}
        if self.ctc_weight > 0:
            ctc_loss = ctc_objf(ctc_enc,
                                tgt_pad,
                                enc_len,
                                egs["tgt_len"],
                                blank=self.ctc_blank,
                                add_softmax=True)
            stats["@ctc"] = ctc_loss.item()
            stats["xent"] = xent_loss.item()
        else:
            ctc_loss = 0
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * xent_loss
        # compute accu
        accu, den = compute_accu(outs, tgts)
        # check coding error
        assert den == egs["#tok"]
        # add to reporter
        stats["accu"] = accu
        stats["loss"] = loss
        return stats


@ApsRegisters.task.register("transducer")
class TransducerTask(Task):
    """
    For RNNT objective function training.
    Args:
        nnet: AM network
        interface: which RNNT loss api to use (warp_rnnt|warprnnt_pytorch)
        blank: blank ID for RNNT loss computation
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
            src_pad: N x Ti x F
            src_len: N
            tgt_pad: N x To
            tgt_len: N
        """
        # tgt_pad: N x To (replace ignore_id with blank)
        tgt_pad, _ = prep_asr_label(egs["tgt_pad"],
                                    egs["tgt_len"],
                                    pad_value=self.blank)
        tgt_len = egs["tgt_len"]
        # N x Ti x To+1 x V
        outs, enc_len = self.nnet(egs["src_pad"], egs["src_len"], tgt_pad,
                                  tgt_len)
        rnnt_kwargs = {"blank": self.blank, "reduction": "sum"}
        # add log_softmax if use https://github.com/1ytic/warp-rnnt
        if self.interface == "warp_rnnt":
            outs = tf.log_softmax(outs, -1)
            rnnt_kwargs["gather"] = True
        # compute loss
        loss = self.rnnt_objf(outs, tgt_pad.to(th.int32), enc_len.to(th.int32),
                              tgt_len.to(th.int32), **rnnt_kwargs)
        loss = loss / th.sum(tgt_len)
        return {"loss": loss}


@ApsRegisters.task.register("lm")
class LmXentTask(Task):
    """
    For LM training (Xent loss)
    Args:
        nnet: language model
        bptt_mode: reuse hidden state in previous batch (for BPTT)
    """

    def __init__(self, nnet: nn.Module, bptt_mode: bool = False) -> None:
        super(LmXentTask, self).__init__(nnet,
                                         description="Xent for LM training")
        self.hidden = None
        self.bptt_mode = bptt_mode

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CE loss, egs contains
            src: N x T+1
            tgt: N x T+1
            len: N
        """
        # pred: N x T+1 x V
        if self.bptt_mode:
            if "reset" in egs and egs["reset"]:
                self.hidden = None
            pred, self.hidden = self.nnet(egs["src"], self.hidden)
        else:
            pred, _ = self.nnet(egs["src"], None, egs["len"])
        loss = ce_objf(pred, egs["tgt"])
        accu, den = compute_accu(pred, egs["tgt"])
        # check coding error
        assert den == egs["#tok"]
        # ppl is derived from xent, so we pass loss to it
        stats = {"accu": accu, "loss": loss, "@ppl": loss.item()}
        return stats
