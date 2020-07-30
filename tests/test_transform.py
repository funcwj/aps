#!/usr/bin/env python

# wujian@2020

import pytest

import torch as th
import numpy as np

from torch_complex.tensor import ComplexTensor

from aps.transform.utils import forward_stft, inverse_stft
from aps.loader import read_wav, write_wav
from aps.transform import AsrTransform, EnhTransform, FixedBeamformer, DfTransform


@pytest.mark.parametrize("wav", [read_wav("data/egs1.wav", sr=16000)])
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


@pytest.mark.parametrize("wav", [read_wav("data/egs1.wav", sr=16000)])
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
    assert transform.feats_dim == shape[-1]


@pytest.mark.parametrize("wav", [read_wav("data/egs2.wav", sr=16000)])
@pytest.mark.parametrize("feats,shape",
                         [("spectrogram-log-cmvn-ipd", [1, 366, 257 * 5]),
                          ("ipd", [1, 366, 257 * 4])])
def test_enh_transform(wav, feats, shape):
    transform = EnhTransform(feats=feats,
                             frame_len=512,
                             frame_hop=256,
                             ipd_index="0,1;0,2;0,3;0,4",
                             aug_prob=0.2)
    feats, stft, _ = transform(th.from_numpy(wav[None, ...]),
                               None,
                               norm_obs=True)
    assert feats.shape == th.Size(shape)
    assert th.sum(th.isnan(feats)) == 0
    assert stft.shape == th.Size([1, 5, 257, 366])
    assert transform.feats_dim == shape[-1]


@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("num_channels", [4, 8])
@pytest.mark.parametrize("num_bins", [257, 513])
@pytest.mark.parametrize("num_directions", [8, 16])
def test_fixed_beamformer(batch_size, num_channels, num_bins, num_directions):
    beamformer = FixedBeamformer(num_directions, num_channels, num_bins)
    num_frames = th.randint(50, 100, (1, )).item()
    inp_r = th.rand(batch_size, num_channels, num_bins, num_frames)
    inp_i = th.rand(batch_size, num_channels, num_bins, num_frames)
    inp_c = ComplexTensor(inp_r, inp_i)
    out_b = beamformer(inp_c)
    assert out_b.shape == th.Size(
        [batch_size, num_directions, num_bins, num_frames])
    out_b = beamformer(inp_c, beam=0)
    assert out_b.shape == th.Size([batch_size, num_bins, num_frames])


@pytest.mark.parametrize("num_bins", [257, 513])
@pytest.mark.parametrize("num_doas", [1, 8])
def test_df_transform(num_bins, num_doas):
    num_channels = 7
    batch_size = 4
    transform = DfTransform(num_bins=num_bins,
                            num_doas=num_doas,
                            af_index="1,0;2,0;3,0;4,0;5,0;6,0")
    num_frames = th.randint(50, 100, (1, )).item()
    phase = th.rand(batch_size, num_channels, num_bins, num_frames)
    doa = th.rand(batch_size)
    df = transform(phase, doa)
    if num_doas == 1:
        assert df.shape == th.Size([batch_size, num_bins, num_frames])
    else:
        assert df.shape == th.Size(
            [batch_size, num_doas, num_bins, num_frames])
