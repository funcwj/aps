#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pytest
import torch as th

from aps.libs import aps_asr_nnet
from aps.transform import AsrTransform

asr_transform = AsrTransform(feats="fbank-log-cmvn",
                             frame_len=400,
                             frame_hop=160,
                             window="hamm",
                             center=False,
                             pre_emphasis=0.97,
                             stft_mode="kaldi")

xfmr_enc_cfg = """
enc_type: xfmr
enc_kwargs:
  num_layers: 4
  chunk: 4
  lctx: 4
  proj: conv2d
  proj_kwargs:
    conv_channels: 128
    num_layers: 2
    kernel: [3, 5]
    stride: [2, 3]
    for_streaming: true
  arch_kwargs:
    att_dim: 512
    att_dropout: 0
    feedforward_dim: 1024
    ffn_dropout: 0.2
    nhead: 8
    pre_norm: true
"""
rnn_enc_cfg = """
enc_type: pytorch_rnn
enc_kwargs:
  num_layers: 3
  hidden: 512
  dropout: 0.2
"""
rnn_dec_cfg = """
dec_kwargs:
  embed_size: 512
  jot_dim: 512
  num_layers: 2
  hidden: 512
  dropout: 0.1
"""


def gen_egs(vocab_size, batch_size):
    u = th.randint(10, 20, (1,)).item()
    x_len = th.randint(16000 * 2, 16000 * 3, (batch_size,))
    x_len = x_len.sort(-1, descending=True)[0]
    y_len = th.randint(u // 2, u, (batch_size,))
    y_len[0] = u
    S = x_len.max().item()
    x = th.rand(batch_size, S)
    y = th.randint(0, vocab_size - 1, (batch_size, u))
    return x, x_len, y, y_len, u


@pytest.mark.parametrize("vocab_size", [400])
def test_streaming_ctc(vocab_size):
    nnet_cls = aps_asr_nnet("streaming_asr@ctc")
    cfg = yaml.safe_load(xfmr_enc_cfg)
    ctc = nnet_cls(input_size=80,
                   vocab_size=vocab_size,
                   asr_transform=asr_transform,
                   enc_type=cfg["enc_type"],
                   enc_kwargs=cfg["enc_kwargs"])
    x, x_len, _, _, _ = gen_egs(vocab_size, 4)
    y = ctc(x, x_len)[0]
    assert th.isnan(y).sum().item() == 0
    assert y.shape[-1] == vocab_size


@pytest.mark.parametrize("vocab_size", [400])
@pytest.mark.parametrize("enc_cfg", [xfmr_enc_cfg, rnn_enc_cfg])
def test_streaming_transducer(vocab_size, enc_cfg):
    nnet_cls = aps_asr_nnet("streaming_asr@transducer")
    enc_cfg = yaml.safe_load(enc_cfg)
    dec_cfg = yaml.safe_load(rnn_dec_cfg)
    rnnt = nnet_cls(input_size=80,
                    vocab_size=vocab_size,
                    asr_transform=asr_transform,
                    enc_type=enc_cfg["enc_type"],
                    enc_proj=512,
                    enc_kwargs=enc_cfg["enc_kwargs"],
                    dec_kwargs=dec_cfg["dec_kwargs"])
    x, x_len, y, y_len, u = gen_egs(vocab_size, 4)
    _, z, _ = rnnt(x, x_len, y, y_len)
    assert z.shape[2:] == th.Size([u + 1, vocab_size])


if __name__ == "__main__":
    # test_streaming_transducer(100, xfmr_enc_cfg)
    test_streaming_ctc(400)
