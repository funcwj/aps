#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import pytest
import librosa

import torch as th
import numpy as np

from torch_complex import ComplexTensor
from aps.transform.utils import forward_stft, inverse_stft
from aps.loader import read_audio
from aps.transform import AsrTransform, EnhTransform, FixedBeamformer, DfTransform
from aps.transform.asr import SpeedPerturbTransform

egs1_wav = read_audio("data/transform/egs1.wav", sr=16000)
egs2_wav = read_audio("data/transform/egs2.wav", sr=16000)


@pytest.mark.parametrize("wav", [egs1_wav])
@pytest.mark.parametrize("frame_len, frame_hop", [(512, 256), (1024, 256),
                                                  (256, 128)])
@pytest.mark.parametrize("window", ["hamm", "sqrthann"])
@pytest.mark.parametrize("center", [True, False])
def test_forward_inverse_stft(wav, frame_len, frame_hop, window, center):
    wav = th.from_numpy(wav)
    mid = forward_stft(wav[None, ...],
                       frame_len,
                       frame_hop,
                       window=window,
                       center=center)
    out = inverse_stft(mid, frame_len, frame_hop, window=window, center=center)
    trunc = min(out.shape[-1], wav.shape[-1])
    th.testing.assert_allclose(out[..., :trunc], wav[..., :trunc])


@pytest.mark.parametrize("wav", [egs1_wav, egs2_wav[0].copy()])
@pytest.mark.parametrize("frame_len, frame_hop", [(512, 256), (1024, 256),
                                                  (400, 160)])
@pytest.mark.parametrize("window", ["hann", "hamm"])
@pytest.mark.parametrize("center", [False, True])
def test_with_librosa(wav, frame_len, frame_hop, window, center):
    real, imag = forward_stft(th.from_numpy(wav)[None, ...],
                              frame_len,
                              frame_hop,
                              window=window,
                              center=center,
                              output="complex")

    torch_mag = (real**2 + imag**2)**0.5
    librosa_mag = np.abs(
        librosa.stft(wav,
                     n_fft=2**math.ceil(math.log2(frame_len)),
                     hop_length=frame_hop,
                     win_length=frame_len,
                     window=window,
                     center=center)).astype("float32")
    librosa_mag = th.from_numpy(librosa_mag)
    th.testing.assert_allclose(torch_mag[0], librosa_mag)


@pytest.mark.parametrize("wav", [egs1_wav])
@pytest.mark.parametrize("feats,shape", [("spectrogram-log", [1, 807, 257]),
                                         ("emph-fbank-log-cmvn", [1, 807, 80]),
                                         ("mfcc", [1, 807, 13]),
                                         ("mfcc-aug", [1, 807, 13]),
                                         ("mfcc-splice", [1, 807, 39]),
                                         ("mfcc-aug-delta", [1, 807, 39])])
def test_asr_transform(wav, feats, shape):
    transform = AsrTransform(feats=feats,
                             frame_len=400,
                             frame_hop=160,
                             use_power=True,
                             pre_emphasis=0.96,
                             aug_prob=0.5,
                             aug_mask_zero=False)
    feats, _ = transform(th.from_numpy(wav[None, ...]), None)
    assert feats.shape == th.Size(shape)
    assert th.sum(th.isnan(feats)) == 0
    assert transform.feats_dim == shape[-1]


@pytest.mark.parametrize("max_length", [160000])
def test_speed_perturb(max_length):
    for _ in range(4):
        speed_perturb = SpeedPerturbTransform(sr=16000)
        wav_len = th.randint(max_length // 2, max_length, (1,))
        wav_out = speed_perturb(th.randn(1, wav_len.item()))
        out_len = speed_perturb.output_length(wav_len)
        assert wav_out.shape[-1] == out_len.item()


@pytest.mark.parametrize("wav", [egs2_wav])
@pytest.mark.parametrize("feats,shape",
                         [("spectrogram-log-cmvn-aug-ipd", [1, 366, 257 * 5]),
                          ("ipd", [1, 366, 257 * 4])])
def test_enh_transform(wav, feats, shape):
    transform = EnhTransform(feats=feats,
                             frame_len=512,
                             frame_hop=256,
                             ipd_index="0,1;0,2;0,3;0,4",
                             aug_prob=0.2)
    feats, stft, _ = transform(th.from_numpy(wav[None, ...]), None)
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
    num_frames = th.randint(50, 100, (1,)).item()
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
    num_frames = th.randint(50, 100, (1,)).item()
    phase = th.rand(batch_size, num_channels, num_bins, num_frames)
    doa = th.rand(batch_size)
    df = transform(phase, doa)
    if num_doas == 1:
        assert df.shape == th.Size([batch_size, num_bins, num_frames])
    else:
        assert df.shape == th.Size([batch_size, num_doas, num_bins, num_frames])


def debug_visualize_feature():
    transform = AsrTransform(feats="fbank-log-aug",
                             frame_len=400,
                             frame_hop=160,
                             use_power=True,
                             aug_prob=1,
                             aug_max_frame=100,
                             aug_max_bands=40,
                             aug_mask_zero=False)
    feats, _ = transform(th.from_numpy(egs1_wav[None, ...]), None)
    import matplotlib.pyplot as plt
    plt.imshow(feats[0].numpy().T)
    plt.show()


def debug_speed_perturb():
    from aps.loader import write_audio
    speed_perturb = SpeedPerturbTransform(sr=16000)
    for i in range(12):
        egs2 = speed_perturb(th.from_numpy(egs1_wav[None, :]))
        write_audio(f"egs1-{i + 1}.wav", egs2[0].numpy())


if __name__ == "__main__":
    debug_speed_perturb()
