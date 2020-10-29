# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from itertools import permutations


def multiple_objf(inp, ref, objf, weight=None, transform=None, batchmean=False):
    """
    Compute summary of multiple loss functions
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (function): function to compute single pair loss (per mini-batch)
    Return:
        loss (Tensor): N (per mini-batch) if batchmean == False
    """
    if len(inp) != len(ref):
        raise ValueError("Size mismatch between #inp and " +
                         f"#ref: {len(inp)} vs {len(ref)}")
    num_tasks = len(inp)
    if weight == None:
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


def permu_invarint_objf(inp,
                        ref,
                        objf,
                        transform=None,
                        batchmean=False,
                        return_permutation=False):
    """
    Compute permutation-invariant loss
    Args:
        inp (list(Object)): estimated list
        ref (list(Object)): reference list
        objf (function): function to compute single pair loss (per mini-batch)
    Return:
        loss (Tensor): N (per mini-batch) if batchmean == False
    """
    if len(inp) != len(ref):
        raise ValueError("Size mismatch between #inp and " +
                         f"#ref: {len(inp)} vs {len(ref)}")

    def perm_objf(permute, out, ref):
        """
            Return tensor (P x N) for each permutation and mini-batch
            """
        return sum([objf(out[s], ref[t]) for s, t in enumerate(permute)
                   ]) / len(permute)

    if transform:
        inp = [transform(i) for i in inp]
        ref = [transform(r) for r in ref]

    loss_mat = th.stack(
        [perm_objf(p, inp, ref) for p in permutations(range(len(inp)))])

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
                inp,
                ref,
                objf,
                weight=None,
                transform=None,
                batchmean=False):
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
                inp,
                ref,
                objf,
                transform=None,
                return_permutation=False,
                batchmean=False):
        return permu_invarint_objf(inp,
                                   ref,
                                   objf,
                                   transform=transform,
                                   return_permutation=return_permutation,
                                   batchmean=batchmean)
