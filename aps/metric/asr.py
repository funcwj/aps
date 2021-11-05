#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
from itertools import permutations
from typing import List, Tuple

import edit_distance as ed


def format_str(str1: str, str2: str) -> Tuple[str]:
    """
    Padding two strings to same length
    """
    delta_len = len(str1) - len(str2)
    if delta_len == 0:
        return str1, str2
    lpad = delta_len // 2
    rpad = delta_len - lpad
    if delta_len < 0:
        return " " * (-lpad) + str1 + " " * (-rpad), str2
    else:
        return str1, " " * lpad + str2 + " " * rpad


def print_operations(hyp: List[str], ref: List[str], ops: List[Tuple]):
    """
    Print operations between hypothesis and text references
    """
    hyp_after_op = []
    ref_after_op = []
    for op_stats in ops:
        op, n1, _, n3, _ = op_stats
        h = hyp[n1]
        r = ref[n3]
        if op == "insert":
            a, b = "*" * len(r), r
        elif op == "delete":
            a, b = h, "*" * len(h)
        else:
            # equal or replace
            a, b = format_str(h, r)
        hyp_after_op.append(a)
        ref_after_op.append(b)
    print("hyp: " + " ".join(hyp_after_op))
    print("ref: " + " ".join(ref_after_op), flush=True)


def wer(hyp: List[str], ref: List[str], details: bool = False) -> Tuple[float]:
    """
    Compute edit distance between two str list
    Args:
        hlist: list[str], hypothesis
        rlist: list[str], references
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
    if details:
        print_operations(hyp, ref, ops)
    if sub_err + del_err + ins_err != error:
        raise RuntimeError("Bugs: sub_err + del_err + ins_err != #error")
    if equal != match:
        raise RuntimeError("Bugs: equal != #match")
    return (sub_err, ins_err, del_err)


def permute_wer(hlist: List[List[str]],
                rlist: List[List[str]],
                details: bool = False) -> Tuple[float]:
    """
    Compute edit distance between N pairs
    Args:
        hlist: list[list[str]], hypothesis
        rlist: list[list[str]], reference
    Return:
        float: three error types (sub/ins/del)
    """

    def distance(hlist, rlist, details):
        # [(sub/ins/del), ..., (sub/ins/del)]
        err_pair = [wer(h, r, details=details) for h, r in zip(hlist, rlist)]
        # (sub/ins/del)
        err = tuple(sum([p[i] for p in err_pair]) for i in range(3))
        return sum(err), err

    N = len(hlist)
    if N != len(rlist):
        raise RuntimeError("size do not match between hlist " +
                           f"and rlist: {N} vs {len(rlist)}")
    if N != 1:
        details = False
    errs = []
    best = math.inf
    pair = -1
    for index, order in enumerate(permutations(range(N))):
        err, permu_errs = distance(hlist, [rlist[n] for n in order], details)
        errs.append(permu_errs)
        if err < best:
            best = err
            pair = index
    return errs[pair]
