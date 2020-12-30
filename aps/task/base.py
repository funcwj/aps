# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch.nn as nn
from typing import Optional


class Task(nn.Module):
    """
    The class that glues the network forward and loss computation
    Args:
        nnet: network instance
        ctx: context for loss computation
        description (str): description for current task instance
    """

    def __init__(self,
                 nnet: nn.Module,
                 ctx: Optional[nn.Module] = None,
                 description: str = "unknown") -> None:
        super(Task, self).__init__()
        self.nnet = nnet
        self.ctx = ctx
        self.description = description
