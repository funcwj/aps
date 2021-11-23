#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn as nn

from aps.libs import dynamic_importlib, ApsRegisters, ApsModules
from aps.conf import load_dict
from aps.asr.transformer.impl import ApsMultiheadAttention
from aps.asr.transformer.utils import digit_shift, prep_sub_mask
from aps.asr.base.attention import padding_mask

external_dir = "tests/data/external"
checkpoint_dir = "tests/data/checkpoint"


@pytest.mark.parametrize(
    "str_lib",
    [f"{external_dir}/nnet.py:VoiceFilter", f"{external_dir}/task.py:DpclTask"])
def test_import_lib(str_lib):
    dynamic_importlib(str_lib)


@pytest.mark.parametrize("str_dict", [
    f"{checkpoint_dir}/aishell_att_1a/dict",
    f"{checkpoint_dir}/timit_rnnt_1a/dict"
])
def test_load_dict(str_dict):
    load_dict(str_dict)
    load_dict(str_dict, reverse=True)


@pytest.mark.parametrize(
    "package",
    ["asr", "streaming_asr", "sse", "task", "loader", "trainer", "transform"])
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


@pytest.mark.parametrize("index", [0, 1, 2])
def test_aps_selfattn(index):
    S, L, N, E = 100, 100, 8, 256
    self_attn = ApsMultiheadAttention(E, 4, dropout=0)
    self_attn.train()
    query = th.rand(L, N, E)
    if index == 0:
        key, value = query, query
    elif index == 1:
        key = th.rand(S, N, E)
        value = key
    else:
        key = th.rand(S, N, E)
        value = th.rand(S, N, E)

    key_len = th.randint(S // 2, S, (N,))
    key_len[0] = S
    key_padding_mask = padding_mask(key_len)
    attn_mask = prep_sub_mask(S)

    my1, my2 = self_attn(query,
                         key,
                         value,
                         None,
                         key_padding_mask=key_padding_mask,
                         attn_mask=attn_mask)
    th1, th2 = self_attn.torch_forward(query,
                                       key,
                                       value,
                                       key_padding_mask=key_padding_mask,
                                       attn_mask=attn_mask)
    assert my1.shape == th1.shape
    assert my2.shape == th2.shape
    th.testing.assert_allclose(my2, th2)
    th.testing.assert_allclose(my1, th1)
