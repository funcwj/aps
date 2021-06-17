#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th

from aps.libs import aps_sse_nnet
from aps.transform import EnhTransform


@pytest.mark.parametrize("num_spks,nonlinear", [
    pytest.param(1, "sigmoid"),
    pytest.param(2, "softmax"),
    pytest.param(2, "relu")
])
def test_base_rnn(num_spks, nonlinear):
    nnet_cls = aps_sse_nnet("sse@base_rnn")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    base_rnn = nnet_cls(enh_transform=transform,
                        num_bins=257,
                        input_size=257,
                        input_proj=512,
                        num_layers=2,
                        hidden=512,
                        num_spks=num_spks,
                        output_nonlinear=nonlinear)
    inp = th.rand(2, 64000)
    x = base_rnn(inp)
    if num_spks > 1:
        x = x[0]
    assert x.shape == th.Size([2, 257, 249])
    z = base_rnn.infer(inp[0])
    if num_spks > 1:
        z = z[0]
    assert z.shape == th.Size([64000])


def test_crn():
    nnet_cls = aps_sse_nnet("sse@crn")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=320,
                             frame_hop=160,
                             round_pow_of_two=False)
    crn = nnet_cls(161,
                   enh_transform=transform,
                   mode="masking",
                   training_mode="freq")
    inp = th.rand(4, 64000)
    x = crn(inp)
    assert x.shape == th.Size([4, 161, 399])
    z = crn.infer(inp[1])
    assert z.shape == th.Size([64000])


def test_phasen():
    nnet_cls = aps_sse_nnet("sse@phasen")
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
    assert y.shape == th.Size([4, 257, 249])
    z = phasen.infer(inp[1])
    assert z.shape == th.Size([64000])


@pytest.mark.parametrize("num_branch", [1, 2])
@pytest.mark.parametrize("cplx", [True, False])
def test_dcunet(num_branch, cplx):
    nnet_cls = aps_sse_nnet("sse@dcunet")
    transform = EnhTransform(feats="", frame_len=512, frame_hop=256)
    dcunet = nnet_cls(enh_transform=transform,
                      K="7,5;7,5;5,3;5,3;3,3;3,3",
                      S="2,1;2,1;2,1;2,1;2,1;2,1",
                      C="32,32,64,64,64,128",
                      P="1,1,1,1,1,0",
                      O="0,0,1,1,1,0",
                      num_branch=num_branch,
                      cplx=cplx,
                      causal_conv=False,
                      freq_padding=True,
                      connection="cat")
    inp = th.rand(4, 64000)
    x = dcunet(inp)
    if num_branch > 1:
        x = x[0]
    assert x.shape == th.Size([4, 64000])
    y = dcunet.infer(inp[1])
    if num_branch > 1:
        y = y[0]
    assert y.shape == th.Size([64000])


@pytest.mark.parametrize("num_spks", [1, 2])
@pytest.mark.parametrize("non_linear", ["", "sigmoid"])
def test_dense_unet(num_spks, non_linear):
    nnet_cls = aps_sse_nnet("sse@dense_unet")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    dense_unet = nnet_cls(K="3,3;3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                          S="1,1;2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                          P="0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1;0,1",
                          O="0,0,0,0,0,0,0,0",
                          enc_channel="16,32,32,32,32,64,128,384",
                          dec_channel="32,16,32,32,32,32,64,128",
                          conv_dropout=0.3,
                          num_spks=num_spks,
                          rnn_hidden=512,
                          rnn_layers=2,
                          rnn_resize=384,
                          rnn_bidir=False,
                          rnn_dropout=0.2,
                          num_dense_blocks=5,
                          enh_transform=transform,
                          non_linear=non_linear,
                          inp_cplx=True,
                          out_cplx=True,
                          training_mode="time")
    inp = th.rand(4, 64000)
    x = dense_unet(inp)
    if num_spks > 1:
        x = x[0]
    assert x.shape == th.Size([4, 64000])
    y = dense_unet.infer(inp[1])
    if num_spks > 1:
        y = y[0]
    assert y.shape == th.Size([64000])


@pytest.mark.parametrize("num_spks", [1, 2])
def test_freq_xfmr_rel(num_spks):
    nnet_cls = aps_sse_nnet("sse@freq_xfmr_rel")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    pose_kwargs = {"radius": 256, "dropout": 0.1}
    arch_kwargs = {
        "att_dropout": 0.1,
        "feedforward_dim": 512,
        "pre_norm": False,
        "att_dim": 256,
        "nhead": 4
    }
    xfmr = nnet_cls(arch="xfmr",
                    input_size=257,
                    enh_transform=transform,
                    num_spks=num_spks,
                    num_bins=257,
                    arch_kwargs=arch_kwargs,
                    pose_kwargs=pose_kwargs,
                    num_layers=3,
                    mask_dropout=0.1,
                    non_linear="sigmoid",
                    training_mode="time")
    inp = th.rand(4, 64000)
    x = xfmr(inp)
    if num_spks > 1:
        assert len(x) == num_spks
        assert x[0].shape == th.Size([4, 64000])
    else:
        assert x.shape == th.Size([4, 64000])
    y = xfmr.infer(inp[1])
    if num_spks > 1:
        y = y[0]
    assert y.shape == th.Size([64000])


@pytest.mark.parametrize("num_spks,nonlinear", [
    pytest.param(1, "sigmoid"),
    pytest.param(2, "softmax"),
    pytest.param(2, "relu")
])
def test_tasnet(num_spks, nonlinear):
    nnet_cls = aps_sse_nnet("sse@time_tasnet")
    tasnet = nnet_cls(L=40,
                      N=256,
                      X=6,
                      R=2,
                      B=256,
                      H=512,
                      P=3,
                      norm="cLN",
                      num_spks=num_spks,
                      non_linear=nonlinear,
                      skip_residual=True,
                      causal=False,
                      mixture_consistency=True)
    inp = th.rand(4, 64000)
    x = tasnet(inp)
    if num_spks > 1:
        x = x[0]
    assert x.shape == th.Size([4, 64000])
    y = tasnet.infer(inp[1])
    if num_spks > 1:
        y = y[0]
    assert y.shape == th.Size([64000])


def test_dprnn():
    nnet_cls = aps_sse_nnet("sse@time_dprnn")
    dprnn = nnet_cls(num_spks=1,
                     input_norm="cLN",
                     block_type="dp",
                     conv_kernels=16,
                     conv_filters=64,
                     proj_filters=64,
                     chunk_len=100,
                     num_layers=2,
                     rnn_hidden=64,
                     rnn_bi_inter=True,
                     non_linear="relu")
    inp = th.rand(4, 64000)
    x = dprnn(inp)
    assert x.shape == th.Size([4, 64000])
    y = dprnn.infer(inp[1])
    assert y.shape == th.Size([64000])


@pytest.mark.parametrize("num_spks", [1, 2])
@pytest.mark.parametrize("cplx", [True, False])
def test_dccrn(num_spks, cplx):
    nnet_cls = aps_sse_nnet("sse@dccrn")
    transform = EnhTransform(feats="spectrogram", frame_len=512, frame_hop=256)
    dccrn = nnet_cls(enh_transform=transform,
                     cplx=cplx,
                     K="3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                     S="2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                     P="1,1,1,1,1,0,0",
                     O="0,0,0,0,0,0,1",
                     C="16,32,64,64,128,128,256",
                     num_spks=num_spks,
                     rnn_resize=512 if cplx else 256,
                     non_linear="sigmoid",
                     connection="cat")
    inp = th.rand(4, 64000)
    x = dccrn(inp)
    if num_spks > 1:
        x = x[0]
    assert x.shape == th.Size([4, 64000])
    y = dccrn.infer(inp[1])
    if num_spks > 1:
        y = y[0]
    assert y.shape == th.Size([64000])


@pytest.mark.parametrize("num_bins", [257])
def test_rnn_enh_ml(num_bins):
    nnet_cls = aps_sse_nnet("sse@rnn_enh_ml")
    transform = EnhTransform(feats="spectrogram-log-cmvn-ipd",
                             frame_len=512,
                             frame_hop=256,
                             ipd_index="0,1;0,2;0,3")
    rnn_enh_ml = nnet_cls(enh_transform=transform,
                          num_bins=num_bins,
                          input_size=num_bins * 4,
                          input_proj=512,
                          num_layers=2,
                          hidden=512)
    inp = th.rand(2, 5, 64000)
    x, y = rnn_enh_ml(inp)
    assert x.shape == th.Size([2, 5, num_bins, 249])
    assert th.isnan(x.real).sum() + th.isnan(x.imag).sum() == 0
    assert y.shape == th.Size([2, 249, num_bins])
    z = rnn_enh_ml.infer(inp[0])
    assert z.shape == th.Size([249, num_bins])


@pytest.mark.parametrize("resampling_factor", [1, 2, 4])
@pytest.mark.parametrize("chunk_len", [16000, 32000])
def test_demucs(resampling_factor, chunk_len):
    nnet_cls = aps_sse_nnet("sse@demucs")
    from aps.sse.enh.demucs import workout_train_chunk_length
    chunk_len_for_train = workout_train_chunk_length(
        chunk_len, resampling_factor=resampling_factor)
    demucs = nnet_cls(resampling_factor=resampling_factor)
    x = th.rand(2, chunk_len_for_train)
    y = demucs(x)
    assert y.shape == th.Size([2, chunk_len_for_train])
    x = th.rand(chunk_len)
    y = demucs.infer(x)
    assert y.shape == th.Size([chunk_len])
