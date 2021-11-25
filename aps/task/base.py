# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch.nn as nn

from typing import Optional

from aps.utils import get_logger
from aps.eval.wrapper import load_checkpoint

logger = get_logger(__name__)


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


class TsTask(Task):
    """
    Base class for TS (Teacher-Student) task
    Args:
        nnet: network instance
        cpt: checkpoint directory for teacher network
        description (str): description for current task instance
    """

    def __init__(self,
                 nnet: nn.Module,
                 cpt: str,
                 cpt_tag: str = "best",
                 description: str = "unknown") -> None:
        super(TsTask, self).__init__(nnet, description=description)
        stats = load_checkpoint(cpt, cpt_tag=cpt_tag)
        self.teacher = stats["nnet"]
        # fix teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Load the checkpoint from {cpt}, epoch: " +
                    f"{stats['epoch']}, tag: {cpt_tag}")
