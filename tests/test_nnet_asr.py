#!/usr/bin/env python

# wujian@2020

import pytest
import torch as th

from aps.asr import support_nnet
from aps.transform import AsrTransform, EnhTransform

default_rnn_decoder_kwargs = {
    "dec_rnn": "lstm",
    "rnn_layers": 2,
    "rnn_hidden": 512,
    "rnn_dropout": 0.1,
    "input_feeding": True,
    "vocab_embeded": True
}

default_xfmr_decoder_kwargs = {
    "att_dim": 512,
    "nhead": 8,
    "feedforward_dim": 2048,
    "pos_dropout": 0,
    "att_dropout": 0.1,
    "num_layers": 2
}

default_rnn_encoder_kwargs = {
    "rnn": "lstm",
    "rnn_layers": 3,
    "rnn_hidden": 512,
    "rnn_dropout": 0.2,
    "rnn_bidir": False
}

custom_rnn_encoder_kwargs = {
    "rnn": "lstm",
    "rnn_layers": 2,
    "rnn_bidir": True,
    "rnn_dropout": 0.2,
    "rnn_hidden": 512,
    "rnn_project": 512,
    "layernorm": True
}

tdnn_encoder_kwargs = {
    "dim": 512,
    "norm": "BN",
    "num_layers": 5,
    "stride": "2,2,2,1,1",
    "dilation": "1,1,1,1,1",
    "dropout": 0.2
}

fsmn_encoder_kwargs = {
    "project_size": 512,
    "num_layers": 45,
    "residual": True,
    "lctx": 10,
    "rctx": 10,
    "norm": "BN",
    "dropout": 0.2
}

tdnn_rnn_encoder_kwargs = {
    "tdnn_dim": 512,
    "tdnn_layers": 3,
    "tdnn_stride": "2,2,2",
    "tdnn_dilation": "1,1,2",
    "rnn": "lstm",
    "rnn_layers": 3,
    "rnn_bidir": True,
    "rnn_dropout": 0.2,
    "rnn_hidden": 320,
}

tdnn_fsmn_encoder_kwargs = {
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
}

transformer_encoder_kwargs = {
    "input_embed": "conv2d",
    "embed_other_opts": -1,
    "att_dim": 512,
    "nhead": 8,
    "feedforward_dim": 2048,
    "pos_dropout": 0.1,
    "att_dropout": 0.1,
    "post_norm": True,
    "num_layers": 2
}

transformer_rel_encoder_kwargs = {
    "input_embed": "conv2d",
    "embed_other_opts": -1,
    "att_dim": 512,
    "nhead": 8,
    "feedforward_dim": 2048,
    "pos_dropout": 0.1,
    "att_dropout": 0.1,
    "post_norm": True,
    "num_layers": 2
}


def gen_egs(vocab_size, batch_size, num_channels=1):
    u = th.randint(10, 20, (1,)).item()
    x_len = th.randint(16000, 16000 * 5, (batch_size,)).sort(-1,
                                                             descending=True)[0]
    S = x_len.max().item()
    if num_channels == 1:
        x = th.rand(batch_size, S)
    else:
        x = th.rand(batch_size, num_channels, S)
    y = th.randint(0, vocab_size - 1, (batch_size, u))
    return x, x_len, y, u


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
                       encoder_type="common_rnn",
                       encoder_proj=256,
                       encoder_kwargs=default_rnn_encoder_kwargs,
                       decoder_dim=512,
                       decoder_kwargs=default_rnn_decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size)
    z, _, _, _ = att_asr(x, x_len, y)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("att_type,att_kwargs", [
    pytest.param("ctx", {"att_dim": 512}),
    pytest.param("mhctx", {
        "att_dim": 512,
        "att_head": 4
    })
])
def test_mvdr_att(att_type, att_kwargs):
    nnet_cls = support_nnet("mvdr_att")
    vocab_size = 100
    batch_size = 4
    num_channels = 4
    mask_net_kwargs = {
        "rnn": "lstm",
        "rnn_layers": 2,
        "rnn_hidden": 512,
        "rnn_dropout": 0.2,
        "rnn_bidir": False,
        "non_linear": "sigmoid"
    }
    mvdr_kwargs = {"att_dim": 512, "mask_norm": True, "eps": 1e-5}
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
                            mvdr_kwargs=mvdr_kwargs,
                            mask_net_kwargs=mask_net_kwargs,
                            asr_transform=asr_transform,
                            enh_transform=enh_transform,
                            att_type=att_type,
                            att_kwargs=att_kwargs,
                            encoder_type="common_rnn",
                            encoder_proj=256,
                            encoder_kwargs=default_rnn_encoder_kwargs,
                            decoder_dim=512,
                            decoder_kwargs=default_rnn_decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size, num_channels=num_channels)
    z, _, _, _ = mvdr_att_asr(x, x_len, y)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("mode,enh_kwargs", [
    pytest.param(
        "clp", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "spectra_complex": True
        }),
    pytest.param(
        "ti", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "apply_log": True
        }),
    pytest.param(
        "ti_att", {
            "num_bins": 257,
            "batchnorm": True,
            "num_channels": 4,
            "spatial_filters": 5,
            "spectra_filters": 128,
            "apply_log": True
        })
])
def test_beam_att(mode, enh_kwargs):
    nnet_cls = support_nnet("beam_att")
    vocab_size = 100
    batch_size = 4
    num_channels = 4
    enh_transform = EnhTransform(feats="",
                                 frame_len=512,
                                 frame_hop=256,
                                 window="sqrthann")
    beam_att_asr = nnet_cls(vocab_size=vocab_size,
                            asr_input_size=640 if mode != "ti_att" else 128,
                            sos=0,
                            eos=1,
                            ctc=True,
                            mode=mode,
                            enh_kwargs=enh_kwargs,
                            asr_transform=None,
                            enh_transform=enh_transform,
                            att_type="dot",
                            att_kwargs={"att_dim": 512},
                            encoder_type="common_rnn",
                            encoder_proj=256,
                            encoder_kwargs=default_rnn_encoder_kwargs,
                            decoder_dim=512,
                            decoder_kwargs=default_rnn_decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size, num_channels=num_channels)
    z, _, _, _ = beam_att_asr(x, x_len, y)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("custom_rnn", custom_rnn_encoder_kwargs),
    pytest.param("tdnn", tdnn_encoder_kwargs),
    pytest.param("fsmn", fsmn_encoder_kwargs),
    pytest.param("tdnn_rnn", tdnn_rnn_encoder_kwargs),
    pytest.param("tdnn_fsmn", tdnn_fsmn_encoder_kwargs)
])
def test_common_encoder(enc_type, enc_kwargs):
    nnet_cls = support_nnet("att")
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
                       att_type="ctx",
                       att_kwargs={"att_dim": 512},
                       encoder_type=enc_type,
                       encoder_proj=256,
                       encoder_kwargs=enc_kwargs,
                       decoder_dim=512,
                       decoder_kwargs=default_rnn_decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size)
    z, _, _, _ = att_asr(x, x_len, y)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("custom_rnn", custom_rnn_encoder_kwargs),
    pytest.param("tdnn", tdnn_encoder_kwargs),
    pytest.param("fsmn", fsmn_encoder_kwargs),
    pytest.param("tdnn_rnn", tdnn_rnn_encoder_kwargs),
    pytest.param("tdnn_fsmn", tdnn_fsmn_encoder_kwargs),
    pytest.param("transformer", transformer_encoder_kwargs),
    pytest.param("transformer_rel", transformer_rel_encoder_kwargs)
])
def test_transformer_encoder(enc_type, enc_kwargs):
    nnet_cls = support_nnet("transformer")
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
        encoder_type=enc_type,
        encoder_proj=512 if "transformer" not in enc_type else None,
        encoder_kwargs=enc_kwargs,
        decoder_type="transformer",
        decoder_kwargs=default_xfmr_decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size)
    z, _, _, _ = xfmr_asr(x, x_len, y)
    assert z.shape == th.Size([4, u + 1, vocab_size - 1])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("custom_rnn", custom_rnn_encoder_kwargs),
    pytest.param("tdnn", tdnn_encoder_kwargs),
    pytest.param("fsmn", fsmn_encoder_kwargs),
    pytest.param("tdnn_rnn", tdnn_rnn_encoder_kwargs),
    pytest.param("tdnn_fsmn", tdnn_fsmn_encoder_kwargs),
    pytest.param("transformer", transformer_encoder_kwargs),
    pytest.param("transformer_rel", transformer_rel_encoder_kwargs)
])
def test_common_transducer(enc_type, enc_kwargs):
    nnet_cls = support_nnet("common_transducer")
    vocab_size = 100
    batch_size = 4
    decoder_kwargs = {
        "embed_size": 512,
        "enc_dim": 512,
        "jot_dim": 512,
        "dec_rnn": "lstm",
        "dec_layers": 2,
        "dec_hidden": 512,
        "dec_dropout": 0.1
    }
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    rnnt = nnet_cls(input_size=80,
                    vocab_size=vocab_size,
                    blank=vocab_size - 1,
                    asr_transform=asr_transform,
                    encoder_type=enc_type,
                    encoder_proj=None if "transformer" in enc_type else 512,
                    encoder_kwargs=enc_kwargs,
                    decoder_kwargs=decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size)
    y_len = th.randint(u // 2, u, (batch_size,))
    z, _ = rnnt(x, x_len, y, y_len)
    assert z.shape[2:] == th.Size([u + 1, vocab_size])


@pytest.mark.parametrize("enc_type,enc_kwargs", [
    pytest.param("custom_rnn", custom_rnn_encoder_kwargs),
    pytest.param("tdnn", tdnn_encoder_kwargs),
    pytest.param("fsmn", fsmn_encoder_kwargs),
    pytest.param("tdnn_rnn", tdnn_rnn_encoder_kwargs),
    pytest.param("tdnn_fsmn", tdnn_fsmn_encoder_kwargs),
    pytest.param("transformer", transformer_encoder_kwargs),
    pytest.param("transformer_rel", transformer_rel_encoder_kwargs)
])
def test_transformer_transducer(enc_type, enc_kwargs):
    nnet_cls = support_nnet("transformer_transducer")
    vocab_size = 100
    batch_size = 4
    decoder_kwargs = {
        "jot_dim": 512,
        "att_dim": 512,
        "nhead": 8,
        "feedforward_dim": 2048,
        "pos_dropout": 0.1,
        "att_dropout": 0.1,
        "num_layers": 2
    }
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    xfmr_rnnt = nnet_cls(input_size=80,
                         vocab_size=vocab_size,
                         blank=vocab_size - 1,
                         asr_transform=asr_transform,
                         encoder_type=enc_type,
                         encoder_proj=512,
                         encoder_kwargs=enc_kwargs,
                         decoder_kwargs=decoder_kwargs)
    x, x_len, y, u = gen_egs(vocab_size, batch_size)
    y_len = th.randint(u // 2, u, (batch_size,))
    y_len[0] = u
    z, _ = xfmr_rnnt(x, x_len, y, y_len)
    assert z.shape[2:] == th.Size([u + 1, vocab_size])
