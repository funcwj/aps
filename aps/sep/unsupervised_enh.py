#!/usr/bin/env python

# wujian@2020

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from aps.asr.base.encoder import TorchRNNEncoder
from scipy.optimize import linear_sum_assignment

EPSILON = th.finfo(th.float32).eps

supported_plan = {
    257: [[20, 70, 170], [2, 90, 190], [2, 50, 150], [2, 110, 210],
          [2, 30, 130], [2, 130, 230], [2, 0, 110], [2, 150, 257]],
    513: [[20, 100, 200], [2, 120, 220], [2, 80, 180], [2, 140, 240],
          [2, 60, 160], [2, 160, 260], [2, 40, 140], [2, 180, 280],
          [2, 0, 120], [2, 200, 300], [2, 220, 320], [2, 240, 340],
          [2, 260, 360], [2, 280, 380], [2, 300, 400], [2, 320, 420],
          [2, 340, 440], [2, 360, 460], [2, 380, 480], [2, 400, 513]]
}


def norm_observation(mat, axis=-1, eps=EPSILON):
    """
    L2 normalization for observation vectors
    """
    denorm = np.linalg.norm(mat, axis=axis, keepdims=True)
    denorm = np.maximum(denorm, eps)
    return mat / denorm


def permu_aligner(masks, transpose=False):
    """
    Solve permutation problems for clustering based mask algorithm
    Reference: "https://raw.githubusercontent.com/fgnt/pb_bss/master/pb_bss/permutation_alignment.py"
    Args:
        masks: K x T x F
    Return:
        aligned_masks: K x T x F
    """
    if masks.ndim != 3:
        raise RuntimeError("Expect 3D TF-masks, K x T x F or K x F x T")
    if transpose:
        masks = np.transpose(masks, (0, 2, 1))
    K, _, F = masks.shape
    # normalized masks, for cos distance, K x T x F
    feature = norm_observation(masks, axis=1)
    # K x F
    mapping = np.stack([np.ones(F, dtype=np.int) * k for k in range(K)])

    if F not in supported_plan:
        raise ValueError(f"Unsupported num_bins: {F}")
    for itr, beg, end in supported_plan[F]:
        for _ in range(itr):
            # normalized centroid, K x T
            centroid = np.mean(feature[..., beg:end], axis=-1)
            centroid = norm_observation(centroid, axis=-1)
            go_on = False
            for f in range(beg, end):
                # K x K
                score = centroid @ norm_observation(feature[..., f], axis=-1).T
                # derive permutation based on score matrix
                index, permu = linear_sum_assignment(score, maximize=True)
                # not ordered
                if np.sum(permu != index) != 0:
                    feature[..., f] = feature[permu, :, f]
                    mapping[..., f] = mapping[permu, f]
                    go_on = True
            if not go_on:
                break
    # K x T x F
    permu_masks = np.zeros_like(masks)
    for f in range(F):
        permu_masks[..., f] = masks[mapping[..., f], :, f]
    return permu_masks


class UnsupervisedEnh(TorchRNNEncoder):
    """
    A recurrent network for unsupervised training
    """
    def __init__(self,
                 input_size=257,
                 num_bins=257,
                 input_project=None,
                 enh_transform=None,
                 rnn="lstm",
                 rnn_layers=3,
                 rnn_hidden=512,
                 rnn_dropout=0.2,
                 rnn_bidir=False):
        super(UnsupervisedEnh, self).__init__(input_size,
                                              num_bins,
                                              rnn=rnn,
                                              input_project=input_project,
                                              rnn_layers=rnn_layers,
                                              rnn_hidden=rnn_hidden,
                                              rnn_dropout=rnn_dropout,
                                              rnn_bidir=rnn_bidir,
                                              non_linear="sigmoid")
        self.enh_transform = enh_transform
        if enh_transform is None:
            raise ValueError("enh_transform can not be None")

    def check_args(self, noisy, training=True):
        """
        Check input arguments
        """
        if training and noisy.dim() != 3:
            raise RuntimeError(
                "UnsupervisedEnh expects 3D tensor (training), " +
                f"got {noisy.dim()} instead")
        if not training and noisy.dim() != 2:
            raise RuntimeError(
                "UnsupervisedEnh expects 2D tensor (training), " +
                f"got {noisy.dim()} instead")

    def infer(self, noisy):
        """
        Args
            noisy: C x S
        Return
            masks (Tensor): T x F
        """
        self.check_args(noisy, training=False)
        with th.no_grad():
            noisy = noisy[None, ...]
            _, masks = self.forward(noisy)
            return masks[0]

    def forward(self, noisy):
        """
        Args
            noisy: N x C x S
        Return
            cspec (ComplexTensor): N x C x F x T
            masks (Tensor): N x T x F
        """
        self.check_args(noisy, training=True)
        # feats: N x T x F
        # cspec: N x C x F x T
        feats, cspec, _ = self.enh_transform(noisy, None, norm_obs=True)
        masks, _ = super().forward(feats, None)
        return cspec, masks
