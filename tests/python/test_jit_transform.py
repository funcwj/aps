#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest
import torch as th

from aps.io import read_audio
from aps.transform.utils import forward_stft, export_jit
from aps.transform import AsrTransform

egs1_wav = read_audio("data/transform/egs1.wav", sr=16000)


@pytest.mark.parametrize("wav", [egs1_wav])
@pytest.mark.parametrize("feats", ["fbank-log-cmvn", "perturb-mfcc-aug-delta"])
def test_asr_transform_jit(wav, feats):
    wav = th.from_numpy(wav[None, ...])
    packed = forward_stft(wav,
                          400,
                          160,
                          mode="librosa",
                          window="hamm",
                          pre_emphasis=0.96,
                          center=False,
                          return_polar=False)
    trans = AsrTransform(feats=feats,
                         stft_mode="librosa",
                         window="hamm",
                         frame_len=400,
                         frame_hop=160,
                         use_power=True,
                         pre_emphasis=0.96,
                         center=False,
                         aug_prob=0.5,
                         aug_mask_zero=False)
    trans.eval()
    scripted_trans = th.jit.script(export_jit(trans.transform))
    ref_out = trans(wav, None)[0]
    jit_out = scripted_trans(packed)
    th.testing.assert_allclose(ref_out, jit_out)


if __name__ == "__main__":
    test_asr_transform_jit(egs1_wav, "fbank-log-cmvn")
