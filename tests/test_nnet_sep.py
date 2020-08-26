#!/usr/bin/env python

# wujian@2020

import torch as th

from aps.sep import support_nnet
from aps.transform import EnhTransform


def test_base_rnn():
    nnet_cls = support_nnet("base_rnn")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    base_rnn = nnet_cls(enh_transform=transform,
                        num_bins=257,
                        input_size=257,
                        input_project=512,
                        rnn_layers=2,
                        num_spks=1,
                        rnn_hidden=512)
    inp = th.rand(2, 64000)
    x = base_rnn(inp)
    assert x.shape == th.Size([2, 257, 249])
    z = base_rnn.infer(inp[0])
    assert z.shape == th.Size([64000])


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


def test_dcunet():
    nnet_cls = support_nnet("dcunet")
    transform = EnhTransform(feats="", frame_len=512, frame_hop=256)
    dcunet = nnet_cls(enh_transform=transform,
                      K="7,5;7,5;5,3;5,3;3,3;3,3",
                      S="2,1;2,1;2,1;2,1;2,1;2,1",
                      C="32,32,64,64,64,128",
                      num_branch=1,
                      cplx=True,
                      causal_conv=False,
                      freq_padding=True,
                      connection="cat")
    inp = th.rand(4, 64000)
    x = dcunet(inp)
    assert x.shape == th.Size([4, 64000])
    y = dcunet.infer(inp[1])
    assert y.shape == th.Size([64000])


def test_tasnet():
    nnet_cls = support_nnet("time_tasnet")
    tasnet = nnet_cls(L=40,
                      N=256,
                      X=8,
                      R=4,
                      B=256,
                      H=512,
                      P=3,
                      input_norm="gLN",
                      norm="BN",
                      num_spks=1,
                      non_linear="relu",
                      block_residual=True,
                      causal=False)
    inp = th.rand(4, 64000)
    x = tasnet(inp)
    assert x.shape == th.Size([4, 64000])
    y = tasnet.infer(inp[1])
    assert y.shape == th.Size([64000])


def test_dprnn():
    nnet_cls = support_nnet("time_dprnn")
    dprnn = nnet_cls(num_spks=1,
                     input_norm="cLN",
                     conv_kernels=16,
                     conv_filters=64,
                     proj_filters=64,
                     chunk_len=100,
                     dprnn_layers=2,
                     dprnn_bi_inter=True,
                     dprnn_hidden=64,
                     dprnn_block="dp",
                     non_linear="relu")
    inp = th.rand(4, 64000)
    x = dprnn(inp)
    assert x.shape == th.Size([4, 64000])
    y = dprnn.infer(inp[1])
    assert y.shape == th.Size([64000])


def test_dccrn():
    nnet_cls = support_nnet("dccrn")
    transform = EnhTransform(feats="", frame_len=512, frame_hop=256)
    dccrn = nnet_cls(enh_transform=transform,
                     cplx=True,
                     K="3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                     S="2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                     C="16,32,64,64,128,128,256",
                     num_spks=1,
                     rnn_resize=1536,
                     non_linear="sigmoid",
                     connection="cat")
    inp = th.rand(4, 64000)
    x = dccrn(inp)
    assert x.shape == th.Size([4, 64000])
    y = dccrn.infer(inp[1])
    assert y.shape == th.Size([64000])


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
    assert th.isnan(x.real).sum() + th.isnan(x.imag).sum() == 0
    assert y.shape == th.Size([2, 249, 257])
    z = unsuper_enh.infer(inp[0])
    assert z.shape == th.Size([249, 257])


if __name__ == "__main__":
    test_dprnn()