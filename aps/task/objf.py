# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from itertools import permutations
from typing import List, Any, Callable, Optional
from aps.const import IGNORE_ID, EPSILON


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
        tgts (Tensor): N x L
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
    N, L = tgts.shape
    if outs.shape[1] < L:
        raise ValueError(
            f"#frames({outs.shape[1]}) < #labels({L}), not valid for CTC")
    num_nans = th.isnan(outs).sum().item()
    if num_nans > 0:
        raise ValueError(f"Get {num_nans} NANs in the tensor")
    # add log-softmax, N x T x V => T x N x V
    if add_softmax:
        outs = tf.log_softmax(outs, dim=-1)
    # CTC loss
    loss = tf.ctc_loss(outs.transpose(0, 1),
                       tgts,
                       out_len,
                       tgt_len,
                       blank=blank,
                       reduction="sum",
                       zero_infinity=True)
    loss = loss / (th.sum(tgt_len) if reduction == "mean" else N)
    return loss


def sisnr_objf(x: th.Tensor,
               s: th.Tensor,
               eps: float = EPSILON,
               zero_mean: bool = True,
               non_nagetive: bool = False) -> th.Tensor:
    """
    Computer SiSNR
    Args:
        x (Tensor): separated signal, N x S
        s (Tensor): reference signal, N x S
    Return:
        sisnr (Tensor): N
    """

    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate " +
                           f"si-snr, {x.shape} vs {s.shape}")
    if zero_mean:
        x = x - th.mean(x, dim=-1, keepdim=True)
        s = s - th.mean(s, dim=-1, keepdim=True)
    t = th.sum(x * s, dim=-1,
               keepdim=True) * s / (l2norm(s, keepdim=True)**2 + eps)

    snr_linear = l2norm(t) / (l2norm(x - t) + eps)
    if non_nagetive:
        return 10 * th.log10(1 + snr_linear**2)
    else:
        return 20 * th.log10(eps + snr_linear)


def snr_objf(x: th.Tensor,
             s: th.Tensor,
             eps: float = EPSILON,
             snr_max: float = -1,
             non_nagetive: bool = False) -> th.Tensor:
    """
    Computer SNR
    Args:
        x (Tensor): separated signal, N x S
        s (Tensor): reference signal, N x S
    Return:
        snr (Tensor): N
    """

    def l2norm(mat, keepdim=False):
        return th.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError("Dimention mismatch when calculate " +
                           f"si-snr, {x.shape} vs {s.shape}")
    if snr_max > 0:
        # 30dB => 0.001
        threshold = 10**(-snr_max / 10)
        s_norm = l2norm(s)**2
        x_s_norm = l2norm(x - s)**2
        return 10 * th.log10(s_norm + eps) - 10 * th.log10(threshold * s_norm +
                                                           x_s_norm + eps)
    else:
        snr_linear = l2norm(s) / (l2norm(x - s) + eps)
        if non_nagetive:
            return 10 * th.log10(1 + snr_linear**2)
        else:
            return 20 * th.log10(eps + snr_linear)


def dpcl_objf(net_embed: th.Tensor,
              classes: th.Tensor,
              weights: th.Tensor,
              num_spks: int = 2,
              whitened: bool = False) -> th.Tensor:
    """
    Compute Deep Clustering Loss
    Args:
        net_embed (Tensor): network embeddings, N x FT x D
        classes (Tensor): classes id for each TF-bin, N x F x T
        weights (Tensor): weights for each TF bin, N x F x T
    Return:
        loss (Tensor): DPCL loss for each utterance
    """
    N, F, T = classes.shape
    # encode one-hot: N x FT x 2
    ref_embed = th.zeros([N, F * T, num_spks], device=classes.device)
    ref_embed.scatter_(2, classes.view(N, T * F, 1), 1)

    def affinity(v, y):
        # z: N x D x D
        z = th.bmm(th.transpose(v, 1, 2), y)
        # N
        return th.norm(z, 2, dim=(1, 2))**2

    # reshape vad_mask: N x TF x 1
    weights = weights.view(N, F * T, 1)
    out = net_embed * weights.sqrt()
    ref = ref_embed * weights.sqrt()
    if whitened:
        raise NotImplementedError("Not implemented for whitened = True")
    else:
        loss = affinity(out, out) + affinity(ref, ref) - affinity(out, ref) * 2
    # return loss per-frame
    return loss / T


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

    # no permutation needed
    if num_spks == 1:
        return objf(inp[0], ref[0])

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


def hybrid_permu_objf(out: List[Any],
                      ref: List[Any],
                      objf: Callable,
                      transform: Optional[Callable] = None,
                      weight: Optional[List[float]] = None,
                      permute: bool = True,
                      permu_num_spks: int = 2) -> th.Tensor:
    """
    Return hybrid loss (pair-wise, permutated or pair-wise + permutated)
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (function): function to compute single pair loss (per mini-batch)
        weight (list(float)): weight on each loss value
        permute (bool): use permutation invariant or not
        permu_num_spks (int): number of speakers when computing PIT
    """
    num_branch = len(out)
    if num_branch != len(ref):
        raise RuntimeError(
            f"Got {len(ref)} references but with {num_branch} outputs")

    if permute:
        # N
        loss = permu_invarint_objf(out[:permu_num_spks],
                                   ref[:permu_num_spks],
                                   objf,
                                   transform=transform)
        # add residual loss
        if num_branch > permu_num_spks:
            # warnings.warn(f"#Branch: {num_branch} > #Speaker: {permu_num_spks}")
            num_weight = num_branch - (permu_num_spks - 1)
            if weight is None:
                weight = [1 / num_weight] * num_weight
            other_loss = multiple_objf(out[permu_num_spks:],
                                       ref[permu_num_spks:],
                                       objf,
                                       weight=weight[1:])
            loss = weight[0] * loss + other_loss
    else:
        loss = multiple_objf(out, ref, objf, weight=weight, transform=transform)
    return loss


class DpclObjfComputer(nn.Module):
    """
    A class for computation of the DPCL loss
    """

    def __init__(self):
        super(DpclObjfComputer, self).__init__()

    def forward(self,
                embedding: th.Tensor,
                magnitude_ref: th.Tensor,
                magnitude_mix: th.Tensor,
                mean: bool = True) -> th.Tensor:
        """
        Args:
            embedding (Tensor): network embeddings, N x FT x D
            magnitude_ref (Tensor): magnitude of each speaker, N x F x T x S
            magnitude_mix (Tensor): magnitude of mixture signal, N x F x T
        Return:
            loss (Tensor): loss of each utterance, N
        """
        num_spks = magnitude_ref.shape[-1]
        # classes: N x F x T
        classes = th.argmax(magnitude_ref, -1)
        # weights: N x F x T
        weights = magnitude_mix / th.sum(magnitude_mix, (-1, -2), keepdim=True)
        # loss: N
        loss = dpcl_objf(embedding,
                         classes,
                         weights,
                         num_spks=num_spks,
                         whitened=False)
        return loss.mean() if mean else loss
