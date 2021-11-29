#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Optional
from aps.task.sse import TimeDomainTask


class EendTask(TimeDomainTask):
    """
    Training of the EEND task
    """

    def __init__(self,
                 nnet: nn.Module,
                 num_spks: int = 2,
                 permute: bool = True,
                 weight: Optional[str] = None) -> None:
        super(EendTask, self).__init__(nnet,
                                       num_spks=num_spks,
                                       permute=permute,
                                       weight=weight,
                                       description="EEND training objective")

    def objf(self, out: th.Tensor, ref: th.Tensor) -> th.Tensor:
        """
        Return BCE loss
        """
        # N x T
        loss = tf.binary_cross_entropy_with_logits(th.squeeze(out),
                                                   ref,
                                                   reduction="none")
        return loss.sum(-1)
