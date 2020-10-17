#!/usr/bin/env python

# wujian@2020

import editdistance as ed
from itertools import permutations


def permute_wer(hlist, rlist):
    """
    Compute edit distance between N pairs
    Args:
        hlist: list[vector], hypothesis
        rlist: list[vector], reference
    Return:
        float: WER
    """

    def distance(hlist, rlist):
        return sum([ed.eval(h, r) for h, r in zip(hlist, rlist)])

    N = len(hlist)
    if N != len(rlist):
        raise RuntimeError("size do not match between hlist "
                           "and rlist: {:d} vs {:d}".format(N, len(rlist)))
    wers = []
    for order in permutations(range(N)):
        wers.append(distance(hlist, [rlist[n] for n in order]))
    return min(wers)
