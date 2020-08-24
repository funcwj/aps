#!/usr/bin/env python

# wujian@2020

import math
import torch as th

from torch.nn.utils import clip_grad_norm_

from aps.task import support_task
from aps.sep import support_nnet as support_sep_nnet
from aps.transform import EnhTransform


def toy_rnn(mode):
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    return support_sep_nnet("base_rnn")(enh_transform=transform,
                                        num_bins=257,
                                        input_size=257,
                                        input_project=512,
                                        rnn_layers=2,
                                        num_spks=2,
                                        rnn_hidden=512,
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
        loss, _ = task(egs)
        loss.backward()
        norm = clip_grad_norm_(task.parameters(), 20)
        assert not math.isnan(loss.item())
        assert not math.isnan(norm.item())


def test_wa():
    nnet = toy_rnn("time")
    kwargs = {"permute": True, "num_spks": 2, "objf": "L1"}
    task = support_task("wa", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)

def test_sisnr():
    nnet = toy_rnn("time")
    kwargs = {
        "permute": True,
        "num_spks": 2,
        "non_nagetive": True
    }
    task = support_task("sisnr", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)

def test_snr():
    nnet = toy_rnn("time")
    kwargs = {
        "permute": True,
        "num_spks": 2,
        "non_nagetive": True
    }
    task = support_task("snr", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)


def test_linear_freq_sa():
    nnet = toy_rnn("freq")
    kwargs = {
        "phase_sensitive": True,
        "truncated": 1,
        "permute": True,
        "num_spks": 2,
        "objf": "L2"
    }
    task = support_task("linear_sa", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)

def test_mel_freq_sa():
    nnet = toy_rnn("freq")
    kwargs = {
        "phase_sensitive": True,
        "truncated": 1,
        "permute": True,
        "num_spks": 2,
        "num_mels": 80
    }
    task = support_task("mel_sa", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)

def test_linear_time_sa():
    nnet = toy_rnn("time")
    kwargs = {
        "frame_len": 512,
        "frame_hop": 256,
        "center": False,
        "window": "hann",
        "stft_normalized": False,
        "permute": True,
        "num_spks": 2,
        "objf": "L2"
    }
    task = support_task("time_linear_sa", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)

def test_mel_time_sa():
    nnet = toy_rnn("time")
    kwargs = {
        "frame_len": 512,
        "frame_hop": 256,
        "center": False,
        "window": "hann",
        "stft_normalized": False,
        "permute": True,
        "num_mels": 80,
        "num_spks": 2
    }
    task = support_task("time_mel_sa", nnet, **kwargs)
    egs = gen_egs(2)
    run_epochs(task, egs, 5)


if __name__ == "__main__":
    test_mel_freq_sa()