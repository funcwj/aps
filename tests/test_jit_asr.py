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


@pytest.mark.parametrize("lctx, rctx, chunk", [(3, 1, 1), (3, 3, 1), (3, 0, 1)])
def test_streaming_self_attn(lctx, rctx, chunk):
    N, T, E, H = 2, 5 * chunk, 32, 4
    rel_att = StreamingRelMultiheadAttention(E,
                                             H,
                                             dropout=0.1,
                                             chunk=chunk,
                                             lctx=lctx,
                                             rctx=rctx)
    rel_att.eval()
    lctx_frames, rctx_frames = lctx * chunk, rctx * chunk
    rel_pos = RelPosEncoding(E // H,
                             lradius=lctx_frames,
                             rradius=rctx_frames + (chunk - 1),
                             dropout=0)
    rel_pos.eval()

    chunk_egs = th.rand(T, N, E)
    chunk_pad = tf.pad(chunk_egs, (0, 0, 0, 0, lctx_frames, rctx_frames),
                       "constant", 0)
    TC = T + lctx_frames + rctx_frames
    masks = prep_context_mask(TC, chunk_size=chunk, lctx=lctx, rctx=rctx)
    seq = th.arange(-TC + 1, TC)
    key_rel_pose = rel_pos(seq)
    out_ref = rel_att(chunk_pad,
                      chunk_pad,
                      chunk_pad,
                      key_rel_pose=key_rel_pose,
                      attn_mask=masks)[0]
    if rctx_frames > 0:
        out_ref = out_ref[lctx_frames:-rctx_frames]
    else:
        out_ref = out_ref[lctx_frames:]
    out_jit = []
    seq = th.arange(-lctx_frames, rctx_frames + chunk)
    key_rel_pose = rel_pos(seq)
    for t in range(0, T, chunk):
        c = chunk_pad[t:t + lctx_frames + rctx_frames + chunk]
        c = rel_att.step(c, key_rel_pose)
        out_jit.append(c)
    out_jit = th.cat(out_jit, 0)
    th.testing.assert_allclose(out_ref, out_jit)


if __name__ == "__main__":
    # test_streaming_conv1d(3, 2, 3)
    # test_streaming_conv2d(3, 2, 3, 32)
    # test_streaming_fsmn(4, 1, 3)
    # test_streaming_lstm()
    test_streaming_self_attn(2, 0, 1)
