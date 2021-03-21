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

# for RNNT loss, we have two options:
# https://github.com/HawkAaron/warp-transducer
# https://github.com/1ytic/warp-rnnt
try:
    from warp_rnnt import rnnt_loss as warp_rnnt_objf
except ImportError:
    warp_rnnt_objf = None
try:
    from warprnnt_pytorch import rnnt_loss as warprnnt_pt_objf
except ImportError:
    warprnnt_pt_objf = None

from typing import Tuple, Dict, NoReturn, Optional
from aps.task.base import Task
from aps.task.objf import ce_objf, ls_objf, ctc_objf
from aps.const import IGNORE_ID
from aps.libs import ApsRegisters

__all__ = ["CtcTask", "CtcXentHybridTask", "TransducerTask", "LmXentTask"]


def compute_accu(dec_out: th.Tensor, tgt_pad: th.Tensor) -> Tuple[float]:
    """
    Compute frame-level accuracy
    Args:
        dec_out: N x T, decoder output
        tgt_ref: N x T, padding target labels
    """
    # N x (To+1)
    pred = th.argmax(dec_out.detach(), dim=-1)
    # ignore mask, -1
    mask = (tgt_pad != IGNORE_ID)
    # numerator
    num_correct = th.sum(pred[mask] == tgt_pad[mask]).float()
    # denumerator
    total = th.sum(mask)
    # return pair
    accu = num_correct / total
    return (accu.item(), total.item())


def compute_ctc_accu(enc_out: th.Tensor,
                     tgt_pad: th.Tensor,
                     tgt_len: th.Tensor,
                     blank: int = -1) -> Tuple[float]:
    """
    Compute token accuracy for CTC greedy search sequence
    Args:
        enc_out: N x T, decoder output
        tgt_pad: N x T, padding target labels
    """
    # N x T
    pred = th.argmax(enc_out.detach(), dim=-1)
    # ignore blank
    blk_mask = (pred != blank)
    ctc_len = th.sum(blk_mask, -1)
    num_correct = 0
    for i, p in enumerate(pred):
        cur_tok = p[blk_mask[i]]
        cur_ref = tgt_pad[i]
        # padding
        to_pad = (tgt_len[i] - ctc_len[i]).item()
        if to_pad:
            cur_tok = tf.pad(cur_tok, (0, to_pad))
        num_correct += th.sum(cur_tok == cur_ref[:tgt_len[i]]).float()
    total = th.sum(tgt_len)
    accu = num_correct / total
    return (accu.item(), total.item())


def prep_asr_label(
        tgt_pad: th.Tensor,
        tgt_len: th.Tensor,
        pad_value: int,
        eos_value: int = -1) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """
    Process asr label for loss and accu computation
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


@ApsRegisters.task.register("asr@ctc")
class CtcTask(Task):
    """
    For CTC objective function only
    Args:
        nnet: AM network
        blank: blank id for CTC
        reduction: reduction option applied to the sum of the loss
    """

    def __init__(self,
                 nnet: nn.Module,
                 blank: int = 0,
                 reduction: str = "batchmean") -> None:
        super(CtcTask, self).__init__(
            nnet, description="CTC objective function training for ASR")
        if reduction not in ["mean", "batchmean"]:
            raise ValueError(f"Unsupported reduction option: {reduction}")
        self.reduction = reduction
        self.ctc_blank = blank

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CTC loss, egs contains:
        src_pad: N x Ti x F, src_len: N, tgt_pad: N x To, tgt_len: N
        """
        # ctc_enc: N x T x V
        _, ctc_enc, enc_len = self.nnet(egs["src_pad"], egs["src_len"])
        ctc_loss = ctc_objf(ctc_enc,
                            egs["tgt_pad"],
                            enc_len,
                            egs["tgt_len"],
                            blank=self.ctc_blank,
                            reduction=self.reduction,
                            add_softmax=True)
        accu, den = compute_ctc_accu(ctc_enc,
                                     egs["tgt_pad"],
                                     egs["tgt_len"],
                                     blank=self.ctc_blank)
        # ignore length of eos
        assert den == egs["#tok"] - ctc_enc.shape[0]
        return {"loss": ctc_loss, "accu": accu}


@ApsRegisters.task.register("asr@ctc_xent")
class CtcXentHybridTask(Task):
    """
    For encoder/decoder attention based AM training. (CTC for encoder, Xent for decoder)
    Args:
        nnet: AM network
        blank: blank id for CTC
        reduction: reduction option applied to the sum of the loss
        lsm_factor: label smoothing factor
        lsm_method: label smoothing method (uniform|unigram)
        ctc_weight: CTC weight
        label_count: label count file
    """

    def __init__(self,
                 nnet: nn.Module,
                 blank: int = 0,
                 reduction: str = "batchmean",
                 lsm_factor: float = 0,
                 lsm_method: str = "uniform",
                 ctc_weight: float = 0,
                 label_count: str = "") -> None:
        super(CtcXentHybridTask, self).__init__(
            nnet, description="CTC + Xent multi-task training for ASR")
        if lsm_method == "unigram" and not label_count:
            raise RuntimeError(
                "Missing label_count to use unigram label smoothing")
        if reduction not in ["mean", "batchmean"]:
            raise ValueError(f"Unsupported reduction option: {reduction}")
        self.eos = nnet.eos
        self.reduction = reduction
        self.ctc_blank = blank
        self.ctc_weight = ctc_weight
        self.lsm_factor = lsm_factor
        self.lsm_method = lsm_method
        self.label_count = load_label_count(label_count)

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CTC & Attention loss, egs contains:
        src_pad: N x Ti x F, src_len: N, tgt_pad: N x To, tgt_len: N, ssr: float if needed
        """
        # tgt_pad: N x To (replace ignore_id with eos, used in decoder)
        # tgts: N x To+1 (pad eos, used in loss)
        tgt_pad, tgts = prep_asr_label(egs["tgt_pad"],
                                       egs["tgt_len"],
                                       pad_value=self.eos,
                                       eos_value=self.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        ssr = egs["ssr"] if "ssr" in egs else 0
        outs, ctc_enc, enc_len = self.nnet(egs["src_pad"],
                                           egs["src_len"],
                                           tgt_pad,
                                           egs["tgt_len"],
                                           ssr=ssr)
        # compute loss
        if self.lsm_factor > 0:
            att_loss = ls_objf(outs,
                               tgts,
                               method=self.lsm_method,
                               reduction=self.reduction,
                               lsm_factor=self.lsm_factor,
                               label_count=self.label_count)
        else:
            att_loss = ce_objf(outs, tgts, reduction=self.reduction)

        stats = {}
        if self.ctc_weight > 0:
            ctc_loss = ctc_objf(ctc_enc,
                                egs["tgt_pad"],
                                enc_len,
                                egs["tgt_len"],
                                blank=self.ctc_blank,
                                reduction=self.reduction,
                                add_softmax=True)
            stats["@ctc"] = ctc_loss.item()
            stats["xent"] = att_loss.item()
        else:
            ctc_loss = 0
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss
        # compute accu
        accu, den = compute_accu(outs, tgts)
        # check coding error
        assert den == egs["#tok"]
        # add to reporter
        stats["accu"] = accu
        stats["loss"] = loss
        return stats


@ApsRegisters.task.register("asr@transducer")
class TransducerTask(Task):
    """
    For RNNT objective function training.
    Args:
        nnet: AM network
        interface: which RNNT loss api to use (warp_rnnt|warprnnt_pytorch)
        reduction: reduction option applied to the sum of the loss
        blank: blank ID for RNNT loss computation
    """

    def __init__(self,
                 nnet: nn.Module,
                 interface: str = "warp_rnnt",
                 reduction: str = "batchmean",
                 blank: int = 0) -> None:
        super(TransducerTask,
              self).__init__(nnet,
                             description="RNNT objective function for ASR")
        if reduction not in ["mean", "batchmean"]:
            raise ValueError(f"Unsupported reduction option: {reduction}")
        self.blank = blank
        self.reduction = reduction
        self._setup_rnnt_backend(interface)

    def _setup_rnnt_backend(self, interface: str) -> NoReturn:
        """
        Setup RNNT loss impl in the backend
        """
        api = {
            "warp_rnnt": warp_rnnt_objf,
            "warprnnt_pytorch": warprnnt_pt_objf
        }
        if interface not in api:
            raise ValueError(f"Unsupported RNNT interface: {interface}")
        self.interface = interface
        self.rnnt_objf = api[interface]
        if self.rnnt_objf is None:
            raise RuntimeError(f"import {interface} failed ..., " +
                               "please check python envrionments")

    def forward(self, egs: Dict) -> Dict:
        """
        Compute transducer loss, egs contains:
        src_pad: N x Ti x F, src_len: N, tgt_pad: N x To, tgt_len: N
        """
        # tgt_pad: N x To (replace ignore_id with blank)
        tgt_pad, _ = prep_asr_label(egs["tgt_pad"],
                                    egs["tgt_len"],
                                    pad_value=self.blank)
        tgt_len = egs["tgt_len"]
        # N x Ti x To+1 x V
        _, dec_out, enc_len = self.nnet(egs["src_pad"], egs["src_len"], tgt_pad,
                                        tgt_len)
        rnnt_kwargs = {"blank": self.blank, "reduction": "sum"}
        # add log_softmax if use https://github.com/1ytic/warp-rnnt
        if self.interface == "warp_rnnt":
            dec_out = tf.log_softmax(dec_out, -1)
            rnnt_kwargs["gather"] = True
        # compute loss
        loss = self.rnnt_objf(dec_out,
                              tgt_pad.to(th.int32), enc_len.to(th.int32),
                              tgt_len.to(th.int32), **rnnt_kwargs)
        denorm = th.sum(
            tgt_len) if self.reduction == "mean" else dec_out.shape[0]
        return {"loss": loss / denorm}


@ApsRegisters.task.register("asr@lm")
class LmXentTask(Task):
    """
    For LM training (Xent loss)
    Args:
        nnet: language model
        bptt_mode: reuse hidden state in previous batch (for BPTT)
        reduction: reduction option applied to the sum of the loss
    """

    def __init__(self,
                 nnet: nn.Module,
                 bptt_mode: bool = False,
                 reduction: str = "batchmean") -> None:
        super(LmXentTask, self).__init__(nnet,
                                         description="Xent for LM training")
        if reduction not in ["mean", "batchmean"]:
            raise ValueError(f"Unsupported reduction option: {reduction}")
        self.hidden = None
        self.bptt_mode = bptt_mode
        self.reduction = reduction

    def forward(self, egs: Dict) -> Dict:
        """
        Compute CE loss, egs contains src: N x T+1, tgt: N x T+1, len: N
        """
        # pred: N x T+1 x V
        if self.bptt_mode:
            if "reset" in egs and egs["reset"]:
                self.hidden = None
            pred, self.hidden = self.nnet(egs["src"], self.hidden)
        else:
            pred, _ = self.nnet(egs["src"], None, egs["len"])
        loss = ce_objf(pred, egs["tgt"], reduction=self.reduction)
        accu, den = compute_accu(pred, egs["tgt"])
        # check coding error
        assert den == egs["#tok"]
        # ppl is derived from xent, so we pass loss to it
        ppl = loss if self.reduction == "mean" else loss * pred.shape[0] / den
        stats = {"accu": accu, "loss": loss, "@ppl": ppl.item()}
        return stats
