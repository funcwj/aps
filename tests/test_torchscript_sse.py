#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn as nn

from aps.transform.enh import FeatureTransform as EnhTransform
from aps.libs import aps_sse_nnet

transform = EnhTransform(feats="spectrogram-log-cmvn",
                         frame_len=512,
                         frame_hop=256,
                         round_pow_of_two=True,
                         center=True)
num_bins = transform.feats_dim


def scriped_and_check(export_nnet):
    # disable enh_transform
    export_nnet.enh_transform = None
    export_nnet.eval()
    scripted_nnet = th.jit.script(export_nnet)
    egs_inp = th.rand(1, 20, num_bins) * 10
    ref_out = export_nnet.mask_predict(egs_inp)
    jit_out = scripted_nnet.mask_predict(egs_inp)
    th.testing.assert_allclose(ref_out, jit_out)


@pytest.mark.parametrize("rnn", ["gru", "lstm"])
@pytest.mark.parametrize("non_linear", ["relu", "softmax"])
def test_freq_rnn(rnn, non_linear):
    nnet_cls = aps_sse_nnet("sse@base_rnn")
    export_nnet = nnet_cls(input_proj=256,
                           enh_transform=transform,
                           num_bins=num_bins,
                           rnn=rnn,
                           num_layers=2,
                           hidden=256,
                           bidirectional=True,
                           output_nonlinear=non_linear)
    scriped_and_check(export_nnet)


@pytest.mark.parametrize("causal", [True, False])
def test_freq_tcn(causal):
    nnet_cls = aps_sse_nnet("sse@freq_tcn")
    export_nnet = nnet_cls(N=4,
                           B=4,
                           enh_transform=transform,
                           scaling_param=True,
                           skip_residual=True,
                           num_bins=num_bins,
                           num_spks=1,
                           causal=causal)
    scriped_and_check(export_nnet)


@pytest.mark.parametrize("num_spks", [1, 2])
def test_freq_dfsmn(num_spks):
    nnet_cls = aps_sse_nnet("sse@dfsmn")
    export_nnet = nnet_cls(dim=256,
                           enh_transform=transform,
                           num_layers=3,
                           project=256,
                           num_bins=num_bins,
                           num_spks=2,
                           dropout=0.1)
    scriped_and_check(export_nnet)


@pytest.mark.parametrize("arch", ["xfmr", "cfmr"])
@pytest.mark.parametrize("pose", ["abs", "rel", "xl"])
def test_freq_xfmr(arch, pose):
    pose_kwargs = {"dropout": 0.1}
    arch_kwargs = {
        "att_dropout": 0.1,
        "feedforward_dim": 512,
        "att_dim": 256,
        "nhead": 4
    }
    nnet_cls = aps_sse_nnet("sse@freq_xfmr")
    export_nnet = nnet_cls(arch="cfmr",
                           pose="abs",
                           input_size=num_bins,
                           enh_transform=transform,
                           num_spks=2,
                           num_bins=num_bins,
                           arch_kwargs=arch_kwargs,
                           pose_kwargs=pose_kwargs,
                           num_layers=3,
                           non_linear="sigmoid",
                           training_mode="freq")
    scriped_and_check(export_nnet)
