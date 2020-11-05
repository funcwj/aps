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
                             frame_hop=256)
    return aps_sse_nnet("base_rnn")(enh_transform=transform,
                                    num_bins=257,
                                    input_size=257,
                                    rnn_layers=2,
                                    num_spks=num_spks,
                                    rnn_hidden=256,
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
    task = aps_task("wa", nnet, **kwargs)
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
    task = aps_task("sisnr", nnet, **kwargs)
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
    task = aps_task("snr", nnet, **kwargs)
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
    task = aps_task("linear_sa", nnet, **kwargs)
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
    task = aps_task("mel_sa", nnet, **kwargs)
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
    task = aps_task("time_linear_sa", nnet, **kwargs)
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
    task = aps_task("time_mel_sa", nnet, **kwargs)
    egs = gen_egs(num_branch)
    run_epochs(task, egs, 5)
