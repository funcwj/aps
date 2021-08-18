#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn.functional as tf

import aps.streaming_asr.base.encoder as encoder
from aps.asr.transformer.utils import prep_context_mask
from aps.asr.transformer.pose import RelPosEncoding
from aps.streaming_asr.utils import compute_conv_context
from aps.streaming_asr.transformer.impl import StreamingRelMultiheadAttention
from aps.streaming_asr.transformer.encoder import StreamingTransformerEncoder


def test_streaming_lstm():
    streaming_lstm = encoder.StreamingRNNEncoder(80,
                                                 80,
                                                 rnn="lstm",
                                                 num_layers=3,
                                                 hidden=256,
                                                 dropout=0.1)
    streaming_lstm.eval()
    scripted_nnet = th.jit.script(streaming_lstm)
    print(scripted_nnet)
    N, T = 2, 20
    egs = th.rand(N, T, 80)
    out_ref = scripted_nnet(egs, None)[0]
    out_jit = []
    scripted_nnet.reset()
    for t in range(T):
        out_jit.append(scripted_nnet.step(egs[:, t]))
    out_jit = th.cat(out_jit, 1)
    th.testing.assert_allclose(out_ref, out_jit)


@pytest.mark.parametrize("K, S, L", [(3, 2, 3)])
def test_streaming_conv1d(K, S, L):
    streaming_conv1d = encoder.StreamingConv1dEncoder(80,
                                                      80,
                                                      dim=512,
                                                      norm="BN",
                                                      kernel=K,
                                                      stride=S,
                                                      num_layers=L,
                                                      dropout=0.2)
    streaming_conv1d.eval()
    scripted_nnet = th.jit.script(streaming_conv1d)
    print(scripted_nnet)
    N, T = 2, 50
    egs = th.rand(N, T, 80)
    lctx, rctx, stride = compute_conv_context(L, K, S)
    egs_pad = tf.pad(egs, (0, 0, lctx, rctx), "constant", 0)
    out_ref = scripted_nnet(egs_pad, None)[0]
    ctx = lctx + rctx + 1
    out_jit = []
    for t in range(0, T + 1, stride):
        out_jit.append(scripted_nnet.step(egs_pad[:, t:t + ctx]))
    out_jit = th.cat(out_jit, 1)
    th.testing.assert_allclose(out_ref, out_jit)


@pytest.mark.parametrize("K, S, L, C", [(3, 2, 3, 64)])
def test_streaming_conv2d(K, S, L, C):
    streaming_conv2d = encoder.StreamingConv2dEncoder(80,
                                                      80,
                                                      norm="BN",
                                                      kernel=K,
                                                      stride=S,
                                                      channel=C,
                                                      num_layers=L)
    streaming_conv2d.eval()
    scripted_nnet = th.jit.script(streaming_conv2d)
    print(scripted_nnet)
    N, T = 2, 50
    egs = th.rand(N, T, 80)
    lctx, rctx, stride = compute_conv_context(L, K, S)
    egs_pad = tf.pad(egs, (0, 0, lctx, rctx), "constant", 0)
    out_ref = scripted_nnet(egs_pad, None)[0]
    ctx = lctx + rctx + 1
    out_jit = []
    for t in range(0, T + 1, stride):
        out_jit.append(scripted_nnet.step(egs_pad[:, t:t + ctx]))
    out_jit = th.cat(out_jit, 1)
    th.testing.assert_allclose(out_ref, out_jit)


@pytest.mark.parametrize("L, R, N", [(3, 1, 3), (3, 0, 3)])
def test_streaming_fsmn(L, R, N):
    streaming_fsmn = encoder.StreamingFSMNEncoder(80,
                                                  80,
                                                  dim=256,
                                                  project=256,
                                                  num_layers=N,
                                                  lctx=L,
                                                  rctx=R,
                                                  residual=False,
                                                  norm="BN",
                                                  dropout=0.0)
    streaming_fsmn.eval()
    scripted_nnet = th.jit.script(streaming_fsmn)
    print(scripted_nnet)
    lctx, rctx = L * N, R * N
    N, T = 2, 20
    egs = th.rand(N, T, 80)
    egs_pad = tf.pad(egs, (0, 0, lctx, rctx), "constant", 0)
    out_ref = scripted_nnet(egs_pad, None)[0]
    assert out_ref.shape == egs.shape
    out_jit = []
    scripted_nnet.reset()
    for t in range(T):
        c = egs_pad[:, t:t + lctx + rctx + 1]
        c = scripted_nnet.step(c)
        out_jit.append(c)
    out_jit = th.cat(out_jit, 1)
    th.testing.assert_allclose(out_ref, out_jit)


@pytest.mark.parametrize("lctx, chunk", [(0, 3), (3, 1), (2, 3)])
def test_streaming_mhsa(lctx, chunk):
    N, T, E, H = 2, 10, 32, 4
    rctx = 0
    rel_att = StreamingRelMultiheadAttention(E,
                                             H,
                                             dropout=0,
                                             chunk=chunk,
                                             lctx=lctx)
    rel_att.eval()
    lctx_frames = lctx * chunk
    rel_pos = RelPosEncoding(E // H,
                             lradius=lctx_frames,
                             rradius=chunk - 1,
                             dropout=0)
    rel_pos.eval()

    chunk_egs = th.rand(T, N, E)
    masks = prep_context_mask(T, chunk_size=chunk, lctx=lctx, rctx=rctx)
    seq = th.arange(-T + 1, T)
    key_rel_pose = rel_pos(seq)
    out_ref = rel_att(chunk_egs,
                      chunk_egs,
                      chunk_egs,
                      key_rel_pose=key_rel_pose,
                      attn_mask=masks)[0]
    seq = th.arange(lctx_frames + chunk)
    seq = seq[None, :] - seq[:, None]
    key_rel_pose = rel_pos(seq)
    rel_att.reset()
    for t in range(0, T, chunk):
        end = t + chunk
        # print(f"{t}: {end}")
        c = rel_att.step(chunk_egs[t:end], key_rel_pose)
        th.testing.assert_allclose(c[:chunk], out_ref[t:t + chunk])


@pytest.mark.parametrize("lctx, chunk", [(0, 3), (3, 1), (2, 3)])
def test_streaming_xfmr_linear(lctx, chunk):
    rctx = 0
    proj_kwargs = {"norm": "BN"}
    pose_kwargs = {"lradius": lctx, "rradius": rctx}
    arch_kwargs = {
        "att_dim": 32,
        "nhead": 4,
        "feedforward_dim": 256,
        "att_dropout": 0.1,
        "ffn_dropout": 0.1,
        "pre_norm": False
    }
    N, T, F = 2, 10, 80
    xfmr = StreamingTransformerEncoder("xfmr",
                                       F,
                                       output_proj=F,
                                       num_layers=4,
                                       lctx=lctx,
                                       chunk=chunk,
                                       proj="linear",
                                       proj_kwargs=proj_kwargs,
                                       pose="rel",
                                       pose_kwargs=pose_kwargs,
                                       arch_kwargs=arch_kwargs)
    xfmr.eval()
    egs = th.rand(N, T, F)
    egs_out = xfmr(egs, None)[0]

    scripted_xfmr = th.jit.script(xfmr)
    key_rel_pose = scripted_xfmr.step_pose()
    xfmr.reset()
    for t in range(0, T, chunk):
        end = t + chunk
        c = xfmr.step(egs[:, t:end], key_rel_pose)
        th.testing.assert_allclose(c[:, :chunk], egs_out[:, t:t + chunk])


@pytest.mark.parametrize("lctx, chunk", [(0, 3), (3, 1), (2, 3)])
def test_streaming_cfmr_linear(lctx, chunk):
    rctx = 0
    proj_kwargs = {"norm": "BN"}
    pose_kwargs = {"lradius": lctx, "rradius": rctx}
    arch_kwargs = {
        "att_dim": 32,
        "nhead": 4,
        "feedforward_dim": 256,
        "kernel_size": 15,
        "att_dropout": 0.1,
        "ffn_dropout": 0.1
    }
    N, T, F = 2, 20, 80
    cfmr = StreamingTransformerEncoder("cfmr",
                                       F,
                                       output_proj=F,
                                       num_layers=1,
                                       lctx=lctx,
                                       chunk=chunk,
                                       proj="linear",
                                       proj_kwargs=proj_kwargs,
                                       pose="rel",
                                       pose_kwargs=pose_kwargs,
                                       arch_kwargs=arch_kwargs)
    cfmr.eval()
    egs = th.rand(N, T, F)
    egs_out = cfmr(egs, None)[0]

    scripted_cfmr = th.jit.script(cfmr)
    key_rel_pose = scripted_cfmr.step_pose()
    cfmr.reset()
    for t in range(0, T, chunk):
        end = t + chunk
        print(f"{t}:{end}")
        c = cfmr.step(egs[:, t:end], key_rel_pose)
        th.testing.assert_allclose(c[:, :chunk], egs_out[:, t:t + chunk])


if __name__ == "__main__":
    # test_streaming_conv1d(3, 2, 3)
    # test_streaming_conv2d(3, 2, 3, 32)
    # test_streaming_fsmn(4, 1, 3)
    # test_streaming_lstm()
    # test_streaming_mhsa(2, 2)
    # test_streaming_xfmr_linear(2, 2)
    test_streaming_cfmr_linear(2, 2)
