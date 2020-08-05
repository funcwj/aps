#!/usr/bin/env python

# wujian@2020

import torch as th 

from aps.sep import support_nnet
from aps.transform import EnhTransform


def test_phasen():
    nnet_cls = support_nnet("phasen")
    transform = EnhTransform(feats="", frame_len=512, frame_hop=256)
    phasen = nnet_cls(12,
                      4,
                      enh_transform=transform,
                      num_tsbs=1,
                      num_bins=257,
                      channel_r=5,
                      conv1d_kernel=9,
                      lstm_hidden=256,
                      linear_size=512)
    inp = th.rand(4, 64000)
    x, y = phasen(inp)
    assert x.shape == th.Size([4, 257, 249])
    assert y.shape == th.Size([4, 2, 257, 249])
    z = phasen.infer(inp[1])
    assert z.shape == th.Size([64000])
