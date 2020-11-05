#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import editdistance as ed
from itertools import permutations
from typing import List


def permute_wer(hlist: List[str], rlist: List[str]) -> float:
    """
    Compute edit distance between N pairs
    Args:
        hlist: list[str], hypothesis
        rlist: list[str], reference
    Return:
        float: WER
    """

    def distance(hlist, rlist):
        return sum([ed.eval(h, r) for h, r in zip(hlist, rlist)])

    N = len(hlist)
    if N != len(rlist):
        raise RuntimeError("size do not match between hlist " +
                           f"and rlist: {N} vs {len(rlist)}")
    wers = []
    for order in permutations(range(N)):
        wers.append(distance(hlist, [rlist[n] for n in order]))
    return min(wers)
