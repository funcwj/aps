#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
For Knownledge Distilling task
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from aps.task.sse import hybrid_objf
from aps.task.base import TsTask
from typing import Dict


class SseFreqTsTask(TsTask):
    """
    Frequency domain TS task for SSE models
    """

    def __init__(self,
                 nnet: nn.Module,
                 teacher: str = "",
                 objf: str = "L1",
                 teacher_tag: str = "best",
                 permute: bool = True,
                 num_spks: int = 2):
        super(SseFreqTsTask, self).__init__(
            nnet,
            teacher,
            cpt_tag=teacher_tag,
            description="Frequency domain KD task for SSE models")
        self.permute = permute
        self.num_spks = num_spks
        self.objf_ptr = tf.l1_loss if objf == "L1" else tf.mse_loss

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        L1 or L2
        """
        loss = self.objf_ptr(out, ref, reduction="none")
        return loss.sum(-1)

    def forward(self, egs: Dict) -> Dict:
        """
        Return loss based on teacher's outputs
        """
        mix = egs["mix"]
        # Get reference from teacher
        ref = self.teacher(mix)
        # do separation or enhancement
        out = self.nnet(mix)

        if isinstance(out, th.Tensor):
            out, ref = [out], [ref]
        loss = hybrid_objf(out,
                           ref,
                           self.objf,
                           permute=self.permute,
                           permu_num_spks=self.num_spks)
        return {"loss": th.mean(loss)}
