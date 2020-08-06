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


def test_unsuper_enh():
    nnet_cls = support_nnet("unsupervised_enh")
    transform = EnhTransform(feats="spectrogram-log-cmvn-ipd",
                             frame_len=512,
                             frame_hop=256,
                             ipd_index="0,1;0,2;0,3;0,4")
    unsuper_enh = nnet_cls(enh_transform=transform,
                           num_bins=257,
                           input_size=1285,
                           input_project=512,
                           rnn_layers=2,
                           rnn_hidden=512)
    inp = th.rand(2, 5, 64000)
    x, y = unsuper_enh(inp)
    assert x.shape == th.Size([2, 5, 257, 249])
    assert y.shape == th.Size([2, 249, 257])
    z = unsuper_enh.infer(inp[0])
    assert z.shape == th.Size([249, 257])


if __name__ == "__main__":
    test_unsuper_enh()