#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th

from aps.transform.enh import FeatureTransform as EnhTransform
from aps.libs import aps_sse_nnet

transform = EnhTransform(feats="spectrogram-log-cmvn",
                         frame_len=512,
                         frame_hop=256,
                         round_pow_of_two=True,
                         center=True)
num_bins = transform.feats_dim


def scriped_and_check(export_nnet, is_cplx=False):
    # disable enh_transform
    export_nnet.enh_transform = None
    export_nnet.eval()
    scripted_nnet = th.jit.script(export_nnet)
    if is_cplx:
        egs_inp = th.rand(1, 20, num_bins, 2) * 10
    else:
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


@pytest.mark.parametrize("num_branchs", [1, 2])
def test_freq_dfsmn(num_branchs):
    nnet_cls = aps_sse_nnet("sse@dfsmn")
    export_nnet = nnet_cls(dim=256,
                           enh_transform=transform,
                           num_layers=3,
                           project=256,
                           num_bins=num_bins,
                           num_branchs=2,
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


@pytest.mark.parametrize("num_branch", [1, 2])
@pytest.mark.parametrize("cplx", [True, False])
def test_freq_dcunet(num_branch, cplx):
    nnet_cls = aps_sse_nnet("sse@dcunet")
    export_nnet = nnet_cls(enh_transform=transform,
                           K="7,5;7,5;5,3;5,3;3,3;3,3",
                           S="2,1;2,1;2,1;2,1;2,1;2,1",
                           C="32,32,64,64,64,128",
                           P="1,1,1,1,1,0",
                           O="0,0,1,1,1,0",
                           num_branch=num_branch,
                           cplx=cplx,
                           causal_conv=False,
                           non_linear="tanh" if cplx else "relu",
                           connection="cat")
    export_nnet.forward_stft = None
    export_nnet.inverse_stft = None
    scriped_and_check(export_nnet, is_cplx=True)


@pytest.mark.parametrize("num_spks", [1, 2])
@pytest.mark.parametrize("cplx", [True, False])
def test_freq_dccrn(num_spks, cplx):
    nnet_cls = aps_sse_nnet("sse@dccrn")
    export_nnet = nnet_cls(enh_transform=transform,
                           cplx=cplx,
                           K="3,3;3,3;3,3;3,3;3,3;3,3;3,3",
                           S="2,1;2,1;2,1;2,1;2,1;2,1;2,1",
                           P="1,1,1,1,1,0,0",
                           O="0,0,0,0,0,0,1",
                           C="16,32,64,64,128,128,256",
                           num_spks=num_spks,
                           rnn_resize=512 if cplx else 256,
                           non_linear="tanh" if cplx else "relu",
                           connection="cat")
    export_nnet.forward_stft = None
    export_nnet.inverse_stft = None
    scriped_and_check(export_nnet, is_cplx=True)


if __name__ == "__main__":
    # test_freq_xfmr("xfmr", "abs")
    # test_freq_rnn("gru", "sigmoid")
    test_freq_dfsmn(1)