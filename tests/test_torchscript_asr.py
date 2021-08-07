#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn.functional as tf

import aps.streaming_asr.base.encoder as encoder
from aps.streaming_asr.utils import compute_conv_context


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
        if t == 0:
            c = egs_pad[:, :lctx + rctx + 1]
        else:
            c = egs_pad[:, t + lctx + rctx]
        c = scripted_nnet.step(c)
        out_jit.append(c)
    out_jit = th.cat(out_jit, 1)
    th.testing.assert_allclose(out_ref, out_jit)


if __name__ == "__main__":
    # test_streaming_conv1d(3, 2, 3)
    # test_streaming_conv2d(3, 2, 3, 32)
    test_streaming_fsmn(4, 1, 3)
    # test_streaming_lstm()
