# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from itertools import permutations
from typing import List, Any, Callable, Optional
from aps.const import IGNORE_ID


def ce_objf(outs: th.Tensor,
            tgts: th.Tensor,
            reduction: str = "mean") -> th.Tensor:
    """
    Cross entropy loss function
    Args:
        outs (Tensor): N x T x V
        tgts (Tensor): N x T
        reduction (str): "mean" or "batchmean"
            mean: average on each label
            batchmean: average on each utterance
    Return
        loss (Tensor): (1)
    """
    N, _, V = outs.shape
    # N(To+1) x V
    outs = outs.contiguous().view(-1, V)
    # N(To+1)
    tgts = tgts.view(-1)
    loss = tf.cross_entropy(outs, tgts, ignore_index=IGNORE_ID, reduction="sum")
    K = th.sum(tgts != IGNORE_ID) if reduction == "mean" else N
    return loss / K


def ls_objf(outs: th.Tensor,
            tgts: th.Tensor,
            method: str = "uniform",
            reduction: str = "mean",
            lsm_factor: float = 0.1,
            label_count: Optional[th.Tensor] = None) -> th.Tensor:
    """
    Label smooth loss function (using KL)
    Args:
        outs (Tensor): N x T x V
        tgts (Tensor): N x T
        method (str): label smoothing method
        reduction (str): "mean" or "batchmean"
            mean: average on each label
            batchmean: average on each utterance
        lsm_factor (float): label smooth factor
    Return
        loss (Tensor): (1)
    """
    if method not in ["uniform", "unigram", "temporal"]:
        raise ValueError(f"Unknown label smoothing method: {method}")
    N, _, V = outs.shape
    # NT x V
    outs = outs.contiguous().view(-1, V)
    # NT
    tgts = tgts.view(-1)
    mask = (tgts != IGNORE_ID)
    # M x V
    outs = th.masked_select(outs, mask[:, None]).view(-1, V)
    # M
    tgts = th.masked_select(tgts, mask)
    # M x V
    if method == "uniform":
        dist = th.full_like(outs, lsm_factor / (V - 1))
    elif method == "unigram":
        if label_count.size(-1) != V:
            raise RuntimeError("#label_count do not match with the #vacab_size")
        dist = th.zeros_like(outs)
        # copy to each row
        dist[:] = label_count
        dist = dist.scatter_(1, tgts[:, None], 0)
        # normalize
        dist = dist * lsm_factor / th.sum(dist, -1, keepdim=True)
    else:
        raise NotImplementedError
    dist = dist.scatter_(1, tgts[:, None], 1 - lsm_factor)
    # KL distance
    loss = tf.kl_div(tf.log_softmax(outs, -1), dist, reduction="sum")
    K = th.sum(mask) if reduction == "mean" else N
    return loss / K


def ctc_objf(outs: th.Tensor,
             tgts: th.Tensor,
             out_len: th.Tensor,
             tgt_len: th.Tensor,
             blank: int = 0,
             reduction: str = "mean",
             add_softmax: bool = True) -> th.Tensor:
    """
    PyTorch CTC loss function
    Args:
        outs (Tensor): N x T x V
        tgts (Tensor): N x T
        out_len (Tensor): N
        tgt_len (Tensor): N
        blank (int): blank id for CTC
        reduction (str): "mean" or "batchmean"
            mean: average on each label
            batchmean: average on each utterance
        add_softmax (bool): add softmax before CTC loss or not
    Return
        loss (Tensor): (1)
    """
    N, _ = tgts.shape
    # add log-softmax, N x T x V => T x N x V
    if add_softmax:
        outs = tf.log_softmax(outs, dim=-1).transpose(0, 1)
    # CTC loss
    loss = tf.ctc_loss(outs,
                       tgts,
                       out_len,
                       tgt_len,
                       blank=blank,
                       reduction="sum",
                       zero_infinity=True)
    loss = loss / (th.sum(tgt_len) if reduction == "mean" else N)
    return loss


def multiple_objf(inp: List[Any],
                  ref: List[Any],
                  objf: Callable,
                  weight: Optional[List[float]] = None,
                  transform: Optional[Callable] = None,
                  batchmean: bool = False) -> th.Tensor:
    """
    Compute the summary of multiple loss functions
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (callable): the function to compute single pair loss (per mini-batch)
        weight (list(float)): weight on each loss value
        transform (callable): transform function on inp & ref
        batchmean (bool): return mean value of the loss
    Return:
        loss (Tensor): N (per mini-batch) if batchmean == False
    """
    if len(inp) != len(ref):
        raise ValueError("Size mismatch between #inp and " +
                         f"#ref: {len(inp)} vs {len(ref)}")
    num_tasks = len(inp)
    if weight is None:
        weight = [1 / num_tasks] * num_tasks

    if len(weight) != len(inp):
        raise RuntimeError(
            f"Missing weight ({len(weight)}) for {len(inp)} tasks")
    if transform:
        inp = [transform(i) for i in inp]
        ref = [transform(r) for r in ref]

    loss = [objf(o, r) for o, r in zip(inp, ref)]
    # NOTE: summary not average
    loss = sum([s * l for s, l in zip(weight, loss)])
    if batchmean:
        loss = th.mean(loss)
    return loss


def permu_invarint_objf(inp: List[Any],
                        ref: List[Any],
                        objf: Callable,
                        transform: Optional[Callable] = None,
                        batchmean: bool = False,
                        return_permutation: bool = False) -> th.Tensor:
    """
    Compute permutation-invariant loss
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (function): function to compute single pair loss (per mini-batch)
        transform (callable): transform function on inp & ref
        batchmean (bool): return mean value of the loss
    Return:
        loss (Tensor): N (per mini-batch) if batchmean == False
    """
    num_spks = len(inp)
    if num_spks != len(ref):
        raise ValueError("Size mismatch between #inp and " +
                         f"#ref: {num_spks} vs {len(ref)}")

    def permu_objf(permu, out, ref):
        """
        Return tensor (P x N) for each permutation and mini-batch
        """
        return sum([objf(out[s], ref[t]) for s, t in enumerate(permu)
                   ]) / len(permu)

    if transform:
        inp = [transform(i) for i in inp]
        ref = [transform(r) for r in ref]

    loss_mat = th.stack(
        [permu_objf(p, inp, ref) for p in permutations(range(num_spks))])

    # if we want to maximize the objective, i.e, snr, remember to add negative flag to the objf
    loss, index = th.min(loss_mat, dim=0)
    if batchmean:
        loss = th.mean(loss)
    if return_permutation:
        return loss, index
    else:
        return loss


class MultiObjfComputer(nn.Module):
    """
    A class to compute summary of multiple objective functions
    """

    def __init__(self):
        super(MultiObjfComputer, self).__init__()

    def forward(self,
                inp: List[Any],
                ref: List[Any],
                objf: Callable,
                weight: Optional[List[float]] = None,
                transform: Optional[Callable] = None,
                batchmean: bool = False) -> th.Tensor:
        return multiple_objf(inp,
                             ref,
                             objf,
                             weight=weight,
                             transform=transform,
                             batchmean=batchmean)


class PermuInvarintObjfComputer(nn.Module):
    """
    A class to compute permutation-invariant objective function
    """

    def __init__(self):
        super(PermuInvarintObjfComputer, self).__init__()

    def forward(self,
                inp: List[Any],
                ref: List[Any],
                objf: Callable,
                transform: Optional[Callable] = None,
                batchmean: bool = False,
                return_permutation: bool = False) -> th.Tensor:
        return permu_invarint_objf(inp,
                                   ref,
                                   objf,
                                   transform=transform,
                                   return_permutation=return_permutation,
                                   batchmean=batchmean)
