#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from aps.libs import aps_task, aps_asr_nnet
from aps.transform import AsrTransform


def test_ctc_xent():
    asr_transform = AsrTransform(feats="fbank-log-cmvn",
                                 frame_len=400,
                                 frame_hop=160,
                                 window="hamm")
    nnet_cls = aps_asr_nnet("att")
    vocab_size = 100
    batch_size = 4
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
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=asr_transform,
                       att_type="ctx",
                       att_kwargs={"att_dim": 512},
                       encoder_type="common_rnn",
                       encoder_proj=256,
                       encoder_kwargs=default_rnn_encoder_kwargs,
                       decoder_dim=512,
                       decoder_kwargs=default_rnn_decoder_kwargs)
    task = aps_task("ctc_xent",
                    att_asr,
                    lsm_factor=0.1,
                    ctc_weight=0.2,
                    blank=vocab_size - 1)
    x_len = th.randint(16000, 16000 * 5, (batch_size,)).sort(-1,
                                                             descending=True)[0]
    x = th.rand(4, x_len.max().item())
    U = th.randint(10, 20, (1,)).item()
    y = th.randint(0, vocab_size - 1, (batch_size, U))
    y_len = th.randint(U // 2, U, (batch_size,)).sort(-1, descending=True)[0]
    for n in range(batch_size):
        y[n, y_len[n].item():] = -1
    egs = {
        "src_len": x_len,
        "src_pad": x,
        "tgt_len": y_len,
        "tgt_pad": y,
        "ssr": 0.1
    }
    stats = task(egs)
    assert not th.isnan(stats["loss"])


def test_rnnt():
    # need GPU, so we pass here
    pass


def test_lm_xent():
    nnet_cls = aps_asr_nnet("rnn_lm")
    vocab_size = 100
    batch_size = 4
    rnnlm = nnet_cls(embed_size=256,
                     vocab_size=vocab_size,
                     rnn="lstm",
                     rnn_layers=2,
                     rnn_hidden=512,
                     rnn_dropout=0.2,
                     tie_weights=False)
    task = aps_task("lm", rnnlm)
    U = th.randint(10, 20, (1,)).item()
    x = th.randint(0, vocab_size - 1, (batch_size, U + 1))
    egs = {"src": x[:-1], "tgt": x[1:], "len": None}
    stats = task(egs)
    assert not th.isnan(stats["loss"])


if __name__ == "__main__":
    test_lm_xent()
