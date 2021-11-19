#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pytest
import torch as th

from aps.libs import aps_task, aps_asr_nnet
from aps.transform import AsrTransform
from aps.const import IGNORE_ID

att_enc_cfg = """
enc_type: pytorch_rnn
enc_kwargs:
  rnn: lstm
  num_layers: 2
  hidden: 512
  dropout: 0.2
  bidirectional: false
"""
att_dec_cfg = """
dec_kwargs:
  rnn: lstm
  num_layers: 2
  hidden: 512
  dropout: 0.1
  input_feeding: True
  onehot_embed: False
"""
rnnt_dec_cfg = """
dec_kwargs:
  embed_size: 512
  jot_dim: 512
  rnn: lstm
  num_layers: 2
  hidden: 512
  dropout: 0.2
"""
asr_transform = AsrTransform(feats="fbank-log-cmvn",
                             frame_len=400,
                             frame_hop=160,
                             window="hamm",
                             round_pow_of_two=True,
                             pre_emphasis=0.96,
                             center=False)


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
    return {
        "#utt": batch_size,
        "#tok": th.sum(y_len).item() + batch_size,
        "src_len": x_len,
        "src_pad": x,
        "tgt_len": y_len,
        "tgt_pad": y
    }


def test_ctc_only():
    nnet_cls = aps_asr_nnet("asr@ctc")
    vocab_size = 100
    batch_size = 4
    enc_cfg = yaml.safe_load(att_enc_cfg)
    ctc_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       asr_transform=asr_transform,
                       enc_type=enc_cfg["enc_type"],
                       enc_kwargs=enc_cfg["enc_kwargs"])
    task = aps_task("asr@ctc", ctc_asr, blank=vocab_size - 1)
    egs = gen_asr_egs(batch_size, vocab_size)
    stats = task(egs)
    assert not th.isnan(stats["loss"])


def test_ctc_xent():
    nnet_cls = aps_asr_nnet("asr@att")
    vocab_size = 100
    batch_size = 4
    enc_cfg = yaml.safe_load(att_enc_cfg)
    dec_cfg = yaml.safe_load(att_dec_cfg)
    att_asr = nnet_cls(input_size=80,
                       vocab_size=vocab_size,
                       sos=0,
                       eos=1,
                       ctc=True,
                       asr_transform=asr_transform,
                       att_type="ctx",
                       att_kwargs={"att_dim": 512},
                       enc_type=enc_cfg["enc_type"],
                       enc_proj=256,
                       enc_kwargs=enc_cfg["enc_kwargs"],
                       dec_dim=512,
                       dec_kwargs=dec_cfg["dec_kwargs"])
    task = aps_task("asr@ctc_xent",
                    att_asr,
                    lsm_factor=0.1,
                    ctc_weight=0.1,
                    blank=vocab_size - 1)
    egs = gen_asr_egs(batch_size, vocab_size)
    egs["ssr"] = 0.1
    stats = task(egs)
    assert not th.isnan(stats["loss"])


@pytest.mark.parametrize("rnnt_api", ["warprnnt_pytorch", "torchaudio"])
def test_rnnt(rnnt_api):
    nnet_cls = aps_asr_nnet("asr@transducer")
    vocab_size = 100
    batch_size = 4
    enc_cfg = yaml.safe_load(att_enc_cfg)
    dec_cfg = yaml.safe_load(rnnt_dec_cfg)
    rnnt_asr = nnet_cls(input_size=80,
                        vocab_size=vocab_size,
                        asr_transform=asr_transform,
                        enc_type=enc_cfg["enc_type"],
                        enc_kwargs=enc_cfg["enc_kwargs"],
                        enc_proj=512,
                        dec_kwargs=dec_cfg["dec_kwargs"])
    task = aps_task("asr@transducer",
                    rnnt_asr,
                    blank=vocab_size - 1,
                    interface=rnnt_api)
    egs = gen_asr_egs(batch_size, vocab_size)
    stats = task(egs)
    assert not th.isnan(stats["loss"])


def test_lm_xent():
    nnet_cls = aps_asr_nnet("asr@rnn_lm")
    vocab_size = 100
    batch_size = 4
    rnnlm = nnet_cls(embed_size=256,
                     vocab_size=vocab_size,
                     rnn="lstm",
                     num_layers=2,
                     hidden_size=512,
                     dropout=0.2,
                     tie_weights=False)
    task = aps_task("asr@lm", rnnlm)
    U = th.randint(10, 20, (1,)).item()
    x = th.randint(0, vocab_size - 1, (batch_size, U + 1))
    egs = {
        "#utt": batch_size,
        "#tok": batch_size * U,
        "src": x[:, :-1],
        "tgt": x[:, 1:].contiguous(),
        "len": None
    }
    stats = task(egs)
    assert not th.isnan(stats["loss"])
