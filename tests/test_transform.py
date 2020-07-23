#!/usr/bin/env python

# wujian@2020

import pytest

import torch as th
import numpy as np

from aps.transform.utils import forward_stft, inverse_stft
from aps.loader import read_wav, write_wav
from aps.transform import AsrTransform, EnhTransform

DATA_FILE = "data/egs1.wav"


@pytest.mark.parametrize("wav", [read_wav(DATA_FILE, sr=16000)])
@pytest.mark.parametrize("frame_len, frame_hop", [(512, 256), (1024, 256),
                                                  (256, 128)])
@pytest.mark.parametrize("window", ["hann", "hamm", "sqrthann"])
@pytest.mark.parametrize("center", [True, False])
def test_forward_inverse_stft(wav, frame_len, frame_hop, window, center):
    wav = th.from_numpy(wav)
    mid = forward_stft(wav[None, ...],
                       frame_len,
                       frame_hop,
                       window=window,
                       center=center)
    out = inverse_stft(mid, frame_len, frame_hop, window=window, center=center)
    assert th.sum((out - wav)**2).item() < 1e-5


@pytest.mark.parametrize("wav", [read_wav(DATA_FILE, sr=16000)])
@pytest.mark.parametrize("feats,shape", [("spectrogram-log", [1, 808, 257]),
                                         ("fbank-log-cmvn", [1, 808, 80]),
                                         ("mfcc", [1, 808, 13]),
                                         ("mfcc-aug", [1, 808, 13]),
                                         ("mfcc-splice", [1, 808, 39]),
                                         ("mfcc-aug-delta", [1, 808, 39])])
def test_asr_transform(wav, feats, shape):
    transform = AsrTransform(feats=feats,
                             frame_len=400,
                             frame_hop=160,
                             aug_prob=0.2)
    feats, _ = transform(th.from_numpy(wav[None, ...]), None)
    assert feats.shape == th.Size(shape)
    assert th.sum(th.isnan(feats)) == 0