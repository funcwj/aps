import torch.nn as nn


class Task(nn.Module):
    """
    Warpper for nnet & loss
    """

    def __init__(self, nnet, ctx=None, name="unknown", weight=None):
        super(Task, self).__init__()
        self.nnet = nnet
        self.ctx = ctx
        self.name = name
        self.weight = None if weight is None else list(
            map(float, weight.split(",")))
