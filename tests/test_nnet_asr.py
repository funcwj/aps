#!/usr/bin/env python

# wujian@2020

import pytest
import torch as th

from aps.asr import support_nnet
from aps.transform import AsrTransform

default_rnn_decoder_kwargs = {
    "dec_rnn": "lstm",
    "rnn_layers": 2,
    "rnn_hidden": 512,
    "rnn_dropout": 0.1,
    "input_feeding": True,
    "vocab_embeded": True
}

default_rnn_encoder_kwargs = {
    "rnn": "lstm",
    "rnn_layers": 3,
    "rnn_hidden": 512,
    "rnn_dropout": 0.2,
    "rnn_bidir": False
}

sc_asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                frame_len=400,
                                frame_hop=160,
                                window="hamm")


@pytest.mark.parametrize("att_type,att_kwargs", [
    pytest.param("ctx", {"att_dim": 512}),
    pytest.param("dot", {"att_dim": 512}),
    pytest.param("loc", {
        "att_dim": 512,
        "att_channels": 128,
        "att_kernel": 11
    }),
    pytest.param("mhctx", {
        "att_dim": 512,
        "att_head": 4
    }),
    pytest.param("mhdot", {
        "att_dim": 512,
        "att_head": 4
    }),
    pytest.param("mhloc", {
        "att_dim": 512,
        "att_channels": 128,
        "att_kernel": 11,
        "att_head": 4
    }),
])
def test_att(att_type, att_kwargs):
    nnet_cls = support_nnet("att")
    vocab_size = 100
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=sc_asr_transform,
                       att_type=att_type,
                       att_kwargs=att_kwargs,
                       encoder_type="common_rnn",
                       encoder_proj=256,
                       encoder_kwargs=default_rnn_encoder_kwargs,
                       decoder_dim=512,
                       decoder_kwargs=default_rnn_decoder_kwargs)
    T = th.randint(10, 20, (1, )).item()
    x_len = th.randint(16000, 16000 * 5, (4, )).sort(-1, descending=True)[0]
    S = x_len.max().item()
    x = th.rand(4, S)
    y = th.randint(0, vocab_size - 1, (4, T))
    z, _, _, _ = att_asr(x, x_len, y)
    assert z.shape == th.Size([4, T + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param(
        "custom_rnn", {
            "rnn": "lstm",
            "rnn_layers": 3,
            "rnn_bidir": True,
            "rnn_dropout": 0.2,
            "rnn_hidden": 512,
            "rnn_project": 512,
            "layernorm": True
        }),
    pytest.param(
        "tdnn", {
            "dim": 512,
            "norm": "BN",
            "num_layers": 5,
            "stride": "2,2,2,1,1",
            "dilation": "1,1,1,1,1",
            "dropout": 0.2
        }),
    pytest.param(
        "fsmn", {
            "project_size": 512,
            "num_layers": 5,
            "residual": True,
            "lctx": 10,
            "rctx": 10,
            "norm": "BN",
            "dropout": 0.2
        }),
    pytest.param(
        "tdnn_rnn", {
            "tdnn_dim": 512,
            "tdnn_layers": 3,
            "tdnn_stride": "2,2,2",
            "tdnn_dilation": "1,1,2",
            "rnn": "lstm",
            "rnn_layers": 3,
            "rnn_bidir": True,
            "rnn_dropout": 0.2,
            "rnn_hidden": 320,
        }),
    pytest.param(
        "tdnn_fsmn", {
            "tdnn_dim": 512,
            "tdnn_layers": 3,
            "tdnn_stride": "2,2,2",
            "tdnn_dilation": "1,1,2",
            "fsmn_layers": 3,
            "fsmn_lctx": 10,
            "fsmn_rctx": 10,
            "fsmn_norm": "LN",
            "fsmn_residual": False,
            "fsmn_dilation": 1,
            "fsmn_project": 512,
            "fsmn_dropout": 0.2
        })
])
def test_encoder(enc_type, enc_kwargs):
    nnet_cls = support_nnet("att")
    vocab_size = 100
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=sc_asr_transform,
                       att_type="ctx",
                       att_kwargs={"att_dim": 512},
                       encoder_type=enc_type,
                       encoder_proj=256,
                       encoder_kwargs=enc_kwargs,
                       decoder_dim=512,
                       decoder_kwargs=default_rnn_decoder_kwargs)
    T = th.randint(10, 20, (1, )).item()
    x_len = th.randint(16000, 16000 * 5, (4, )).sort(-1, descending=True)[0]
    S = x_len.max().item()
    x = th.rand(4, S)
    y = th.randint(0, vocab_size - 1, (4, T))
    z, _, _, _ = att_asr(x, x_len, y)
    assert z.shape == th.Size([4, T + 1, vocab_size - 1])