#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pytest
import torch as th
import torch.nn as nn

# import matplotlib.pyplot as plt
from aps.trainer.lr import LrScheduler


def test_optimizer(lr: float = 0.001):
    nnet = nn.Sequential(nn.Linear(50, 100), nn.ReLU())
    return th.optim.Adam(nnet.parameters(), lr=lr, weight_decay=1.0e-5)


@pytest.mark.parametrize("scheduler", ["exp", "linear", "cos", "power"])
def test_warmup_decay(scheduler):
    optimizer = test_optimizer(lr=0.001)
    peak_lr = 0.001
    stop_lr = 1.0e-5
    time_stamps = [1000, 2000, 10000]
    scheduler_cls = LrScheduler[f"warmup_{scheduler}_decay_lr"]
    lr_scheduler = scheduler_cls(optimizer,
                                 time_stamps=time_stamps,
                                 peak_lr=peak_lr,
                                 stop_lr=stop_lr)
    step_lr = []
    optimizer.step()
    for _ in range(10000 + 1000):
        step_lr.append(lr_scheduler.get_lr()[0])
        lr_scheduler.step()
    # plt.plot(step_lr)
    # plt.show()
