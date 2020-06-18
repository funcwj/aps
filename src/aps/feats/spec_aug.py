#!/usr/bin/env python

import random
import torch as th


def tf_mask(batch,
            shape,
            max_bands=30,
            max_frame=40,
            num_freq_masks=2,
            num_time_masks=2,
            device="cpu"):
    """
    Return batch of TF-masks
    Args:
        batch: batch size, N
        shape: (T x F)
    Return:
        masks (Tensor): 0,1 masks, N x T x F
    """
    T, F = shape
    max_frame = min(max_frame, T // 2)
    max_bands = min(max_bands, F // 2)
    mask = []
    for _ in range(batch):
        fmask = random_mask(shape,
                            max_steps=max_bands,
                            num_masks=num_freq_masks,
                            order="freq",
                            device=device)
        tmask = random_mask(shape,
                            max_steps=max_frame,
                            num_masks=num_time_masks,
                            order="time",
                            device=device)
        mask.append(fmask * tmask)
    # N x T x F
    return th.stack(mask)


def random_mask(shape, max_steps=30, num_masks=2, order="freq", device="cpu"):
    """
    Generate random 0/1 masks
    Args:
        shape: (T, F)
    Return:
        masks (Tensor): 0,1 masks, T x F
    """
    if order not in ["time", "freq"]:
        raise RuntimeError(f"Unknown order: {order}")
    # shape: T x F
    masks = th.ones(shape, device=device)
    L = shape[1] if order == "freq" else shape[0]
    for _ in range(num_masks):
        dur = random.randint(1, max_steps - 1)
        if L - dur <= 0:
            continue
        beg = random.randint(0, L - dur - 1)
        if order == "freq":
            masks[:, beg:beg + dur] = 0
        else:
            masks[beg:beg + dur, :] = 0
    return masks