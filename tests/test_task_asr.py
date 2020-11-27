#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th

from aps.libs import aps_task, aps_asr_nnet
from aps.transform import AsrTransform
from aps.const import IGNORE_ID

asr_transform = AsrTransform(feats="fbank-log-cmvn",
                             frame_len=400,
                             frame_hop=160,
                             window="hamm",
                             round_pow_of_two=True,
                             pre_emphasis=0.96,
                             center=False)

att_enc_kwargs = {
    "rnn": "lstm",
    "num_layers": 2,
    "hidden": 512,
    "dropout": 0.2,
    "bidirectional": False
}

att_dec_kwargs = {
    "dec_rnn": "lstm",
    "rnn_layers": 2,
    "rnn_hidden": 512,
    "rnn_dropout": 0.1,
    "input_feeding": True,
    "vocab_embeded": True
}

rnnt_dec_kwargs = {
    "embed_size": 512,
    "jot_dim": 512,
    "dec_rnn": "lstm",
    "dec_layers": 2,
    "dec_hidden": 512,
    "dec_dropout": 0.2
}


def gen_asr_egs(batch_size, vocab_size):
    x_len = th.randint(16000, 16000 * 5, (batch_size,))
    # sort
    x_len = x_len.sort(-1, descending=True)[0]
    x = th.rand(4, x_len.max().item())
    U = th.randint(10, 20, (1,)).item()
    y = th.randint(0, vocab_size - 1, (batch_size, U))
    # sort
    y_len = th.randint(U // 2, U, (batch_size,)).sort(-1, descending=True)[0]
    # make sure first one has no padding
    y_len[0] = U
    for i, n in enumerate(y_len.tolist()):
        y[i, n:] = IGNORE_ID
    return {"src_len": x_len, "src_pad": x, "tgt_len": y_len, "tgt_pad": y}


def test_ctc_xent():
    nnet_cls = aps_asr_nnet("att")
    vocab_size = 100
    batch_size = 4
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=asr_transform,
                       att_type="ctx",
                       att_kwargs={"att_dim": 512},
                       enc_type="vanilla_rnn",
                       enc_proj=256,
                       enc_kwargs=att_enc_kwargs,
                       dec_dim=512,
                       dec_kwargs=att_dec_kwargs)
    task = aps_task("ctc_xent",
                    att_asr,
                    lsm_factor=0.1,
                    ctc_weight=0.2,
                    blank=vocab_size - 1)
    egs = gen_asr_egs(batch_size, vocab_size)
    egs["ssr"] = 0.1
    stats = task(egs)
    assert not th.isnan(stats["loss"])


def test_rnnt():
    nnet_cls = aps_asr_nnet("common_transducer")
    vocab_size = 100
    batch_size = 4
    rnnt_asr = nnet_cls(input_size=80,
                        vocab_size=vocab_size,
                        asr_transform=asr_transform,
                        blank=vocab_size - 1,
                        enc_type="vanilla_rnn",
                        enc_kwargs=att_enc_kwargs,
                        enc_proj=512,
                        dec_kwargs=rnnt_dec_kwargs)
    task = aps_task("transducer",
                    rnnt_asr,
                    blank=vocab_size - 1,
                    interface="warprnnt_pytorch")
    egs = gen_asr_egs(batch_size, vocab_size)
    stats = task(egs)
    assert not th.isnan(stats["loss"])


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
