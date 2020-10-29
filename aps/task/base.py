# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch.nn as nn


class Task(nn.Module):
    """
    Warpper for nnet & loss
    """

    def __init__(self, nnet, ctx=None, name="unknown"):
        super(Task, self).__init__()
        self.nnet = nnet
        self.ctx = ctx
        self.name = name
