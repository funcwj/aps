#!/usr/bin/env python

# wujian@2020

import numpy as np
from itertools import permutations


def si_snr(x, s, eps=1e-8, remove_dc=True):
    """
    Compute Si-SNR
    Args:
        x: vector, enhanced/separated signal
        s: vector, reference signal (ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / (vec_l2norm(s_zm)**2 + eps)
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / (vec_l2norm(s)**2 + eps)
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / (vec_l2norm(n) + eps) + eps)


def permute_si_snr(xlist, slist, align=False):
    """
    Compute Si-SNR between N pairs
    Args:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal (ground truth)
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError("size do not match between xlist " +
                           f"and slist: {N} vs {len(slist)}")
    si_snrs = []
    perm = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
        perm.append(order)
    if not align:
        return max(si_snrs)
    else:
        max_idx = np.argmax(si_snrs)
        return max(si_snrs), perm[max_idx]
