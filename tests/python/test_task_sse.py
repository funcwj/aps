#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import pytest
import torch as th

from torch.nn.utils import clip_grad_norm_
from aps.libs import aps_task, aps_sse_nnet
from aps.transform import EnhTransform


def toy_rnn(mode, num_spks):
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256,
                             center=True,
                             stft_mode="librosa")
    base_rnn_cls = aps_sse_nnet("sse@base_rnn")
    return base_rnn_cls(enh_transform=transform,
                        num_bins=257,
                        input_size=257,
                        num_layers=2,
                        num_spks=num_spks,
                        hidden=256,
                        training_mode=mode)


def gen_egs(num_spks):
    batch_size, chunk_size = 4, 64000
    egs = {
        "mix": th.rand(batch_size, chunk_size),
        "ref": [th.rand(batch_size, chunk_size) for _ in range(num_spks)]
    }
    if num_spks == 1:
        egs["ref"] = egs["ref"][0]
    return egs


def run_epochs(task, egs, iters):
    for _ in range(iters):
        stats = task(egs)
        loss = stats["loss"]
        loss.backward()
        norm = clip_grad_norm_(task.parameters(), 20)
        assert not math.isnan(loss.item())
        assert not math.isnan(norm.item())


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_wa(num_branch, num_spks, permute):
    nnet = toy_rnn("time", num_branch)
    kwargs = {"permute": permute, "num_spks": num_spks, "objf": "L1"}
    task = aps_task("sse@wa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_sisnr(num_branch, num_spks, permute):
    nnet = toy_rnn("time", num_branch)
    kwargs = {"permute": permute, "num_spks": num_spks, "non_nagetive": True}
    task = aps_task("sse@sisnr", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_snr(num_branch, num_spks, permute):
    nnet = toy_rnn("time", num_branch)
    kwargs = {"permute": permute, "num_spks": num_spks, "non_nagetive": True}
    task = aps_task("sse@snr", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_linear_freq_sa(num_branch, num_spks, permute):
    nnet = toy_rnn("freq", num_branch)
    kwargs = {
        "phase_sensitive": True,
        "truncated": 1,
        "permute": permute,
        "num_spks": num_spks,
        "objf": "L2"
    }
    task = aps_task("sse@freq_linear_sa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_mel_freq_sa(num_branch, num_spks, permute):
    nnet = toy_rnn("freq", num_branch)
    kwargs = {
        "phase_sensitive": True,
        "truncated": 1,
        "permute": permute,
        "num_spks": num_spks,
        "num_mels": 80
    }
    task = aps_task("sse@freq_mel_sa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_linear_time_sa(num_branch, num_spks, permute):
    nnet = toy_rnn("time", num_branch)
    kwargs = {
        "frame_len": 512,
        "frame_hop": 256,
        "center": False,
        "window": "hann",
        "stft_normalized": False,
        "permute": permute,
        "num_spks": num_spks,
        "objf": "L2"
    }
    task = aps_task("sse@time_linear_sa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_branch,num_spks,permute", [
    pytest.param(2, 2, True),
    pytest.param(2, 2, False),
    pytest.param(3, 2, True)
])
def test_mel_time_sa(num_branch, num_spks, permute):
    nnet = toy_rnn("time", num_branch)
    kwargs = {
        "frame_len": 512,
        "frame_hop": 256,
        "center": False,
        "window": "hann",
        "stft_normalized": False,
        "permute": permute,
        "num_mels": 80,
        "num_spks": num_spks
    }
    task = aps_task("sse@time_mel_sa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)


@pytest.mark.parametrize("num_spks", [1, 2])
def test_complex_mapping(num_spks):
    nnet_cls = aps_sse_nnet("sse@dense_unet")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256,
                             center=True)
    # output complex spectrogram
    nnet = nnet_cls(K="3,3;3,3;3,3;3,3;3,3;3,3;3,3;3,3",
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
                    num_dense_blocks=2,
                    enh_transform=transform,
                    non_linear="",
                    inp_cplx=True,
                    out_cplx=True,
                    training_mode="freq")
    kwargs = {"objf": "L1", "num_spks": num_spks}
    task = aps_task("sse@complex_mapping", nnet, **kwargs)
    egs = gen_egs(num_spks)
    run_epochs(task, egs, 3)


@pytest.mark.parametrize("num_channels", [3])
def test_enh_ml(num_channels):
    nnet_cls = aps_sse_nnet("sse@rnn_enh_ml")
    transform = EnhTransform(feats="spectrogram-log-cmvn-ipd",
                             frame_len=512,
                             frame_hop=256,
                             ipd_index="0,1;0,2")
    rnn_ml = nnet_cls(enh_transform=transform,
                      num_bins=257,
                      input_size=257 * 3,
                      input_proj=512,
                      num_layers=2,
                      hidden=512)
    task = aps_task("sse@enh_ml", rnn_ml)
    egs = {"mix": th.rand(4, num_channels, 64000)}
    run_epochs(task, egs, 3)
