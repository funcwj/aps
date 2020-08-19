#!/usr/bin/env python

# wujian@2020

import math
import torch as th

from torch.nn.utils import clip_grad_norm_

from aps.task import support_task
from aps.sep import support_nnet as support_sep_nnet
from aps.transform import EnhTransform


def test_linear_time_sa():
    nnet_cls = support_sep_nnet("base_rnn")
    transform = EnhTransform(feats="spectrogram-log-cmvn",
                             frame_len=512,
                             frame_hop=256)
    base_rnn = nnet_cls(enh_transform=transform,
                        num_bins=257,
                        input_size=257,
                        input_project=512,
                        rnn_layers=2,
                        num_spks=2,
                        rnn_hidden=512,
                        training_mode="time")
    kwargs = {
        "frame_len": 512,
        "frame_hop": 256,
        "center": False,
        "window": "hann",
        "stft_normalized": False,
        "permute": True,
        "num_spks": 2,
        "objf": "L2"
    }
    task = support_task("time_linear_sa", base_rnn, **kwargs)
    print(task)
    batch_size, chunk_size = 4, 64000
    for _ in range(5):
        egs = {
            "mix":
            th.zeros(batch_size, chunk_size),
            "ref":
            [th.zeros(batch_size, chunk_size),
             th.rand(batch_size, chunk_size)]
        }
        # with th.autograd.detect_anomaly():
        loss, _ = task(egs)
        loss.backward()
        norm = clip_grad_norm_(task.parameters(), 20)
        assert not math.isnan(loss.item())
        assert not math.isnan(norm.item())

if __name__ == "__main__":
    test_linear_time_sa()