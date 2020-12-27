#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn as nn

from aps.libs import dynamic_importlib, ApsRegisters, ApsModules
from aps.conf import load_dict
from aps.asr.xfmr.pose import digit_shift


@pytest.mark.parametrize(
    "str_lib",
    ["data/external/nnet.py:VoiceFilter", "data/external/task.py:DpclTask"])
def test_import_lib(str_lib):
    dynamic_importlib(str_lib)


@pytest.mark.parametrize("str_dict", ["data/checkpoint/aishell_att_1a/dict"])
def test_load_dict(str_dict):
    load_dict(str_dict)
    load_dict(str_dict, reverse=True)


@pytest.mark.parametrize(
    "package", ["asr", "sse", "task", "loader", "trainer", "transform"])
def test_register(package):
    attr = getattr(ApsModules, package)
    attr.import_all()
    for c in ApsRegisters.container:
        print(c.keys())


@pytest.mark.parametrize("N, H, T, D, K", [
    pytest.param(2, 4, 32, 64, 8),
    pytest.param(2, 4, 32, 64, 32),
    pytest.param(2, 4, 31, 64, 16)
])
def test_rel_pose(N, H, T, D, K):
    # T x N x H x D
    query = th.rand(T, N, H, D)
    emb = nn.Embedding(K * 2 + 1, D)
    pos_vec = th.arange(T)
    rel_mat = th.clamp(pos_vec[None, :] - pos_vec[:, None], max=K, min=-K) + K
    # T x T x D
    rel_mat_emb = emb(rel_mat)
    # T x N x H x T
    ans1 = th.matmul(query, rel_mat_emb[:, None].transpose(-1, -2))
    pos_vec = th.arange(-T + 1, T)
    rel_vec = th.clamp(pos_vec, max=K, min=-K) + K
    # 2T-1 x D
    rel_vec_emb = emb(rel_vec)
    # T x N x H x 2T-1
    ans2 = th.matmul(query, rel_vec_emb.transpose(0, 1))
    # T x N x H x T
    ans2 = digit_shift(ans2)
    th.testing.assert_allclose(ans1, ans2)
