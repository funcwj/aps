#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn

from typing import Optional
from aps.sse.base import SSEBase


class RealTimeSSEBase(SSEBase):
    """
    Base class for real-time speech enhancement/separation
    """

    def __init__(self,
                 transform: Optional[nn.Module],
                 training_mode: str = "freq"):
        super(RealTimeSSEBase, self).__init__(transform,
                                              training_mode=training_mode)

    def step(self, chunk: th.Tensor) -> th.Tensor:
        # use for jit export
        raise NotImplementedError()
