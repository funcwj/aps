#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import edit_distance as ed
from itertools import permutations
from typing import List, Tuple


def wer(hyp: List[str], ref: List[str]) -> Tuple[float]:
    """
    Compute edit distance between two str list
    Args:
        hlist: list[str], hypothesis
        rlist: list[str], reference
    Return:
        float: three error types (sub/ins/del)
    """
    error, match, ops = ed.edit_distance_backpointer(hyp, ref)
    sub_err, ins_err, del_err, equal = 0, 0, 0, 0
    for op in ops:
        if op[0] == "delete":
            del_err += 1
        elif op[0] == "replace":
            sub_err += 1
        elif op[0] == "insert":
            ins_err += 1
        else:
            equal += 1
    if sub_err + del_err + ins_err != error:
        raise RuntimeError("Bugs: sub_err + del_err + ins_err != #error")
    if equal != match:
        raise RuntimeError("Bugs: equal != #match")
    return (sub_err, ins_err, del_err)


def permute_wer(hlist: List[List[str]], rlist: List[List[str]]) -> Tuple[float]:
    """
    Compute edit distance between N pairs
    Args:
        hlist: list[list[str]], hypothesis
        rlist: list[list[str]], reference
    Return:
        float: three error types (sub/ins/del)
    """

    def distance(hlist, rlist):
        # [(sub/ins/del), ..., (sub/ins/del)]
        err_pair = [wer(h, r) for h, r in zip(hlist, rlist)]
        # (sub/ins/del)
        err = tuple(sum([p[i] for p in err_pair]) for i in range(3))
        return sum(err), err

    N = len(hlist)
    if N != len(rlist):
        raise RuntimeError("size do not match between hlist " +
                           f"and rlist: {N} vs {len(rlist)}")
    errs = []
    best = -math.inf
    pair = -1
    for index, order in enumerate(permutations(range(N))):
        err, permu_errs = distance(hlist, [rlist[n] for n in order])
        errs.append(permu_errs)
        if err < best:
            best = err
            pair = index
    return errs[pair]
