#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from typing import Tuple, List, Optional
from torch.optim import lr_scheduler as lr, Optimizer


class CustomMultiStepLR(lr.MultiStepLR):
    """
    Using lr_conf to set milestones & gamma, e.g., 0.1@10,20,40
    """

    def __init__(self,
                 optimizer: Optimizer,
                 lr_conf: str = "0.5@10,20,30,40",
                 last_epoch: int = -1) -> None:
        gamma, milestones = self._parse_args(lr_conf)
        super(CustomMultiStepLR, self).__init__(optimizer,
                                                milestones,
                                                gamma=gamma,
                                                last_epoch=last_epoch)

    def _parse_args(self, lr_conf: str) -> Tuple[float, List[int]]:
        lr_str = lr_conf.split("@")
        if len(lr_str) != 2:
            raise RuntimeError(f"Wrong format for lr_conf={lr_conf}")
        gamma = float(lr_str[0])
        milestones = list(map(int, lr_str[0].split(",")))
        return gamma, milestones


class NoamLR(lr._LRScheduler):
    """
    Lr schuduler for Transformer
    """

    def __init__(self,
                 optimizer: Optimizer,
                 transformer_dim: int = 512,
                 factor: float = 1,
                 warmup: int = 8000,
                 last_epoch: int = -1) -> None:
        self.warmup = warmup
        self.const = factor * transformer_dim**(-0.5)
        super(NoamLR, self).__init__(optimizer, last_epoch=last_epoch)

    def info(self) -> str:
        beg_lr = self.get_lr(1)[0]
        top_lr = self.get_lr(self.warmup)[0]
        return f"learning rate at step 1: {beg_lr:.3e} and " + \
            f"step {self.warmup:d}: {top_lr:.3e}"

    def get_lr(self, step: Optional[int] = None) -> List[float]:
        """
        const = factor * transformer_dim^{-0.5}
        1) cur_step > warmup:   const * cur_step**(-0.5)
        2) cur_step < warmup:   const * cur_step * warmup**(-1.5)
        3) cur_step = warmup:   const * warmup**(-0.5)
        """
        if step is None:
            step = self._step_count
        return [
            self.const * min(step**(-0.5), step * self.warmup**(-1.5))
            for _ in self.optimizer.param_groups
        ]


lr_scheduler_cls = {
    "reduce_lr": lr.ReduceLROnPlateau,
    "step_lr": lr.StepLR,
    "multi_step_lr": CustomMultiStepLR,
    "exponential_lr": lr.ExponentialLR,
    "noam_lr": NoamLR
}
