#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th

from aps.libs import aps_asr_nnet
from aps.transform import AsrTransform, EnhTransform
from aps.asr.base.encoder import Conv1dEncoder, Conv2dEncoder

default_rnn_dec_kwargs = {
    "dec_rnn": "lstm",
    "rnn_layers": 2,
    "rnn_hidden": 512,
    "rnn_dropout": 0.1,
    "input_feeding": True,
    "onehot_embed": False
}

default_xfmr_dec_kwargs = {
    "num_layers": 2,
    "arch_kwargs": {
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "att_dropout": 0.1,
        "ffn_dropout": 0.1
    },
    "pose_kwargs": {
        "scaled": False,
        "dropout": 0
    }
}

default_xfmr_transducer_dec_kwargs = {
    "jot_dim": 512,
    "att_dim": 512,
    "pose_kwargs": {
        "scaled": False,
        "dropout": 0.1
    },
    "arch_kwargs": {
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "ffn_dropout": 0.1,
        "att_dropout": 0.1
    },
    "num_layers": 2
}

default_rnn_enc_kwargs = {
    "rnn": "lstm",
    "num_layers": 3,
    "hidden": 512,
    "dropout": 0.2,
    "bidirectional": False
}

custom_rnn_enc_kwargs = {
    "rnn": "lstm",
    "num_layers": 2,
    "bidirectional": True,
    "dropout": 0.2,
    "hidden": 512,
    "project": 512,
    "norm": "LN"
}

conv1d_enc_kwargs = {
    "dim": 512,
    "norm": "BN",
    "num_layers": 5,
    "stride": [2, 2, 2, 1, 1],
    "dilation": [1, 1, 1, 1, 1],
    "dropout": 0.2
}

fsmn_enc_kwargs = {
    "dim": 512,
    "project": 256,
    "num_layers": 4,
    "residual": True,
    "lcontext": 10,
    "rcontext": 10,
    "norm": "BN",
    "dropout": 0.2
}

conv2d_rnn_enc_kwargs = {
    "conv2d": {
        "out_features": -1,
        "channel": 32,
        "num_layers": 2,
        "stride": 2,
        "kernel": 3,
    },
    "pytorch_rnn": {
        "rnn": "lstm",
        "num_layers": 3,
        "bidirectional": True,
        "dropout": 0.2,
        "hidden": 320
    }
}

conv1d_rnn_enc_kwargs = {
    "conv1d": {
        "out_features": 512,
        "dim": 512,
        "num_layers": 3,
        "stride": [2, 2, 2],
        "dilation": [1, 1, 2],
    },
    "pytorch_rnn": {
        "rnn": "lstm",
        "num_layers": 3,
        "bidirectional": True,
        "dropout": 0.2,
        "hidden": 320
    }
}

conv1d_fsmn_enc_kwargs = {
    "conv1d": {
        "out_features": 512,
        "dim": 512,
        "num_layers": 3,
        "stride": [2, 2, 2],
        "dilation": [1, 1, 2],
    },
    "fsmn": {
        "num_layers": 3,
        "lcontext": 10,
        "rcontext": 10,
        "norm": "LN",
        "residual": False,
        "dilation": 1,
        "project": 512,
        "dropout": 0.2
    }
}

xfmr_abs_enc_kwargs = {
    "proj": "conv2d",
    "proj_kwargs": {
        "conv_channels": 128,
        "num_layers": 3
    },
    "pose": "abs",
    "pose_kwargs": {
        "dropout": 0.1,
    },
    "arch_kwargs": {
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "att_dropout": 0.2,
        "ffn_dropout": 0.3,
        "pre_norm": False,
    },
    "num_layers": 2
}

xfmr_rel_enc_kwargs = {
    "proj": "conv2d",
    "proj_kwargs": {
        "conv_channels": 128,
        "num_layers": 3
    },
    "pose": "rel",
    "pose_kwargs": {
        "radius": 128,
        "dropout": 0.1
    },
    "arch_kwargs": {
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "att_dropout": 0.2,
        "ffn_dropout": 0.3,
        "pre_norm": True
    },
    "num_layers": 2
}

xfmr_xl_enc_kwargs = {
    "proj": "conv2d",
    "proj_kwargs": {
        "conv_channels": 128,
        "num_layers": 3
    },
    "pose": "xl",
    "pose_kwargs": {
        "dropout": 0.1
    },
    "arch_kwargs": {
        "tie": False,
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "att_dropout": 0.2,
        "ffn_dropout": 0.3,
        "pre_norm": True
    },
    "num_layers": 2
}

cfmr_rel_enc_kwargs = {
    "proj": "conv2d",
    "proj_kwargs": {
        "conv_channels": 128,
        "num_layers": 3
    },
    "pose": "rel",
    "pose_kwargs": {
        "radius": 128,
        "dropout": 0.1
    },
    "arch_kwargs": {
        "kernel_size": 11,
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "att_dropout": 0.2,
        "ffn_dropout": 0.3
    },
    "num_layers": 2
}


def gen_egs(vocab_size, batch_size, num_channels=1):
    u = th.randint(10, 20, (1,)).item()
    x_len = th.randint(16000, 16000 * 5, (batch_size,))
    x_len = x_len.sort(-1, descending=True)[0]
    y_len = th.randint(u // 2, u, (batch_size,))
    y_len[0] = u
    S = x_len.max().item()
    if num_channels == 1:
        x = th.rand(batch_size, S)
    else:
        x = th.rand(batch_size, num_channels, S)
    y = th.randint(0, vocab_size - 1, (batch_size, u))
    return x, x_len, y, y_len, u


@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("inp_len", [100, 102])
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv1d_encoder(inp_len, num_layers, dilation):
    conv1d_encoder = Conv1dEncoder(80,
                                   512,
                                   num_layers=num_layers,
                                   stride=2,
                                   dilation=dilation)
    batch_size = 4
    inp = th.rand(batch_size, inp_len, 80)
    out, out_len = conv1d_encoder(inp, th.LongTensor([inp_len] * batch_size))
    assert out.shape[1] == out_len[0]


@pytest.mark.parametrize("num_layers", [2, 3])
@pytest.mark.parametrize("inp_len", [100, 102])
@pytest.mark.parametrize("kernel", [3, 5])
def test_conv2d_encoder(inp_len, num_layers, kernel):
    conv1d_encoder = Conv2dEncoder(80,
                                   -1,
                                   channel=[32] * num_layers,
                                   kernel=kernel,
                                   stride=2,
                                   num_layers=num_layers)
    batch_size = 4
    inp = th.rand(batch_size, inp_len, 80)
    out, out_len = conv1d_encoder(inp, th.LongTensor([inp_len] * batch_size))
    assert out.shape[1] == out_len[0]


@pytest.mark.parametrize("att_type,att_kwargs", [
    pytest.param("ctx", {"att_dim": 512}),
    pytest.param("dot", {"att_dim": 512}),
    pytest.param("loc", {
        "att_dim": 512,
        "conv_channels": 8,
        "loc_context": 100
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
        "conv_channels": 8,
        "loc_context": 64,
        "att_head": 4
    }),
])
def test_asr_att(att_type, att_kwargs):
    nnet_cls = aps_asr_nnet("asr@att")
    vocab_size = 100
    batch_size = 4
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=asr_transform,
                       att_type=att_type,
                       att_kwargs=att_kwargs,
                       enc_type="pytorch_rnn",
                       enc_proj=256,
                       enc_kwargs=default_rnn_enc_kwargs,
                       dec_dim=512,
                       dec_kwargs=default_rnn_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size, batch_size)
    z, _, _ = att_asr(x, x_len, y, y_len)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("att_type,att_kwargs", [
    pytest.param("ctx", {"att_dim": 512}),
    pytest.param("mhctx", {
        "att_dim": 512,
        "att_head": 4
    })
])
def test_mvdr_att(att_type, att_kwargs):
    nnet_cls = aps_asr_nnet("asr@enh_att")
    vocab_size = 100
    batch_size = 4
    num_channels = 4
    enh_kwargs = {
        "rnn": "lstm",
        "num_layers": 2,
        "rnn_inp_proj": 512,
        "hidden_size": 512,
        "dropout": 0.2,
        "bidirectional": False,
        "mvdr_att_dim": 512,
        "mask_norm": True,
        "num_bins": 257
    }
    asr_transform = AsrTransform(feats="abs-mel-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    enh_transform = EnhTransform(feats="spectrogram-log-cmvn-ipd",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm",
                                 ipd_index="0,1;0,2;0,3",
                                 cos_ipd=True)
    mvdr_att_asr = nnet_cls(enh_input_size=257 * 4,
                            vocab_size=vocab_size,
                            sos=0,
                            eos=1,
                            ctc=True,
                            enh_type="rnn_mask_mvdr",
                            enh_kwargs=enh_kwargs,
                            asr_transform=asr_transform,
                            enh_transform=enh_transform,
                            att_type=att_type,
                            att_kwargs=att_kwargs,
                            enc_type="pytorch_rnn",
                            enc_proj=256,
                            enc_kwargs=default_rnn_enc_kwargs,
                            dec_dim=512,
                            dec_kwargs=default_rnn_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size,
                                    batch_size,
                                    num_channels=num_channels)
    z, _, _ = mvdr_att_asr(x, x_len, y, y_len)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enh_type,enh_kwargs", [
    pytest.param(
        "google_clp", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "spectra_complex": True
        }),
    pytest.param(
        "time_invar", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "apply_log": True
        }),
    pytest.param(
        "time_invar_att", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "apply_log": True
        })
])
def test_beam_att(enh_type, enh_kwargs):
    nnet_cls = aps_asr_nnet("asr@enh_att")
    vocab_size = 100
    batch_size = 4
    num_channels = 4
    enh_transform = EnhTransform(feats="",
                                 frame_len=512,
                                 frame_hop=256,
                                 window="sqrthann")
    beam_att_asr = nnet_cls(
        vocab_size=vocab_size,
        asr_input_size=640 if enh_type != "time_invar_att" else 128,
        sos=0,
        eos=1,
        ctc=True,
        enh_type=enh_type,
        enh_kwargs=enh_kwargs,
        asr_transform=None,
        enh_transform=enh_transform,
        att_type="dot",
        att_kwargs={"att_dim": 512},
        enc_type="pytorch_rnn",
        enc_proj=256,
        enc_kwargs=default_rnn_enc_kwargs,
        dec_dim=512,
        dec_kwargs=default_rnn_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size,
                                    batch_size,
                                    num_channels=num_channels)
    z, _, _ = beam_att_asr(x, x_len, y, y_len)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("variant_rnn", custom_rnn_enc_kwargs),
    pytest.param("conv1d", conv1d_enc_kwargs),
    pytest.param("fsmn", fsmn_enc_kwargs),
    pytest.param("concat", conv1d_rnn_enc_kwargs),
    pytest.param("concat", conv1d_fsmn_enc_kwargs),
    pytest.param("concat", conv2d_rnn_enc_kwargs),
    pytest.param("xfmr", xfmr_abs_enc_kwargs),
    pytest.param("xfmr", xfmr_rel_enc_kwargs),
    pytest.param("xfmr", xfmr_xl_enc_kwargs),
    pytest.param("cfmr", cfmr_rel_enc_kwargs)
])
def test_att_encoder(enc_type, enc_kwargs):
    nnet_cls = aps_asr_nnet("asr@att")
    vocab_size = 100
    batch_size = 4
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    att_kwargs = {"att_dim": 512}
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=asr_transform,
                       att_type="ctx",
                       att_kwargs=att_kwargs,
                       enc_type=enc_type,
                       enc_proj=256,
                       enc_kwargs=enc_kwargs,
                       dec_type="rnn",
                       dec_dim=512,
                       dec_kwargs=default_rnn_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size, batch_size)
    z, _, _ = att_asr(x, x_len, y, y_len)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("variant_rnn", custom_rnn_enc_kwargs),
    pytest.param("conv1d", conv1d_enc_kwargs),
    pytest.param("fsmn", fsmn_enc_kwargs),
    pytest.param("concat", conv1d_rnn_enc_kwargs),
    pytest.param("concat", conv1d_fsmn_enc_kwargs),
    pytest.param("concat", conv2d_rnn_enc_kwargs),
    pytest.param("xfmr", xfmr_abs_enc_kwargs),
    pytest.param("xfmr", xfmr_rel_enc_kwargs),
    pytest.param("xfmr", xfmr_xl_enc_kwargs),
    pytest.param("cfmr", cfmr_rel_enc_kwargs)
])
def test_xfmr_encoder(enc_type, enc_kwargs):
    nnet_cls = aps_asr_nnet("asr@xfmr")
    vocab_size = 100
    batch_size = 4
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    xfmr_asr = nnet_cls(
        input_size=80,
        vocab_size=vocab_size,
        sos=0,
        eos=1,
        ctc=True,
        asr_transform=asr_transform,
        enc_type=enc_type,
        enc_proj=512 if enc_type not in ["xfmr", "cfmr"] else None,
        enc_kwargs=enc_kwargs,
        dec_type="xfmr",
        dec_kwargs=default_xfmr_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size, batch_size)
    z, _, _ = xfmr_asr(x, x_len, y, y_len)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("variant_rnn", custom_rnn_enc_kwargs),
    pytest.param("conv1d", conv1d_enc_kwargs),
    pytest.param("fsmn", fsmn_enc_kwargs),
    pytest.param("concat", conv1d_rnn_enc_kwargs),
    pytest.param("concat", conv1d_fsmn_enc_kwargs),
    pytest.param("xfmr", xfmr_abs_enc_kwargs),
    pytest.param("xfmr", xfmr_rel_enc_kwargs),
    pytest.param("xfmr", xfmr_xl_enc_kwargs),
    pytest.param("cfmr", cfmr_rel_enc_kwargs)
])
def test_common_transducer(enc_type, enc_kwargs):
    nnet_cls = aps_asr_nnet("asr@transducer")
    vocab_size = 100
    batch_size = 4
    dec_kwargs = {
        "embed_size": 512,
        "enc_dim": 512,
        "jot_dim": 512,
        "add_ln": True,
        "dec_rnn": "lstm",
        "dec_layers": 2,
        "dec_hidden": 512,
        "dec_dropout": 0.1
    }
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    xfmr_encoders = ["xfmr_abs", "xfmr_rel", "xfmr_xl", "cfmr_xl"]
    rnnt = nnet_cls(input_size=80,
                    vocab_size=vocab_size,
                    asr_transform=asr_transform,
                    enc_type=enc_type,
                    enc_proj=None if enc_type in xfmr_encoders else 512,
                    enc_kwargs=enc_kwargs,
                    dec_kwargs=dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size, batch_size)
    _, z, _ = rnnt(x, x_len, y, y_len)
    assert z.shape[2:] == th.Size([u + 1, vocab_size])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("variant_rnn", custom_rnn_enc_kwargs),
    pytest.param("conv1d", conv1d_enc_kwargs),
    pytest.param("fsmn", fsmn_enc_kwargs),
    pytest.param("concat", conv1d_rnn_enc_kwargs),
    pytest.param("concat", conv1d_fsmn_enc_kwargs),
    pytest.param("xfmr", xfmr_abs_enc_kwargs),
    pytest.param("xfmr", xfmr_rel_enc_kwargs),
    pytest.param("xfmr", xfmr_xl_enc_kwargs),
    pytest.param("cfmr", cfmr_rel_enc_kwargs)
])
def test_xfmr_transducer(enc_type, enc_kwargs):
    nnet_cls = aps_asr_nnet("asr@xfmr_transducer")
    vocab_size = 100
    batch_size = 4
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    xfmr_rnnt = nnet_cls(input_size=80,
                         vocab_size=vocab_size,
                         asr_transform=asr_transform,
                         enc_type=enc_type,
                         enc_proj=512,
                         enc_kwargs=enc_kwargs,
                         dec_kwargs=default_xfmr_transducer_dec_kwargs)
    x, x_len, y, y_len, u = gen_egs(vocab_size, batch_size)
    _, z, _ = xfmr_rnnt(x, x_len, y, y_len)
    assert z.shape[2:] == th.Size([u + 1, vocab_size])
