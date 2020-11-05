#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
# From https://github.com/funcwj/deep-clustering/blob/c42e91944bdbde5653558a96c10d52fc8c366add/trainer.py#L124

import torch as th
import torch.nn as nn


class DpclTask(nn.Module):
    """
    Reference:
    Deep Clustering Discriminative Embeddings for Segmentation and Separation
    """

    def __init__(self, nnet):
        self.nnet = nnet

    def forward(self, egs, **kwargs):
        # N x T x F
        tgt_masks = egs["ref"]
        # N x T x F
        ibm_masks = egs["ibm"]
        # N x TF x D
        net_embed = self.nnet(egs["mix"])

        N, T, F = tgt_masks.shape
        # shape binary_mask: N x TF x 1
        ibm_masks = ibm_masks.view(N, T * F, 1)

        # encode one-hot
        tgt_embed = th.zeros([N, T * F, self.num_spks], device=tgt_masks.device)
        tgt_embed.scatter_(2, tgt_masks.view(N, T * F, 1), 1)

        # net_embed: N x TF x D
        # tgt_embed: N x TF x S
        net_embed = net_embed * ibm_masks
        tgt_embed = tgt_embed * ibm_masks

        l2_loss = lambda x: th.norm(x, 2)**2
        loss = l2_loss(th.bmm(th.transpose(net_embed, 1, 2), net_embed)) + \
            l2_loss(th.bmm(th.transpose(tgt_embed, 1, 2), tgt_embed)) - \
            l2_loss(th.bmm(th.transpose(net_embed, 1, 2), tgt_embed)) * 2

        return loss / th.sum(ibm_masks)
