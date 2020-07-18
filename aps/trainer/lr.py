#!/usr/bin/env python

# wujian@2020

import math

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, ExponentialLR


class CustomMultiStepLR(MultiStepLR):
    """
    Using lr_conf to set milestones & gamma, e.g., 0.1@10,20,40
    """
    def __init__(self,
                 optimizer,
                 lr_conf="0.5@10,20,30,40",
                 last_epoch=-1,
                 mode="min"):
        lr_str = lr_conf.split("@")
        if len(lr_str) != 2:
            raise RuntimeError(f"Wrong format for lr_conf={lr_conf}")
        gamma = float(lr_str[0])
        milestones = list(map(int, lr_str[0].split(",")))
        super(CustomMultiStepLR, self).__init__(optimizer,
                                                milestones,
                                                gamma=gamma,
                                                last_epoch=last_epoch)
        self.best = math.inf if mode == "min" else -math.inf
        self.mode = mode

    def step(self, value):
        if value < self.best and self.mode == "min":
            self.best = value
        if value > self.best and self.mode == "max":
            self.best = value
        super().step()


class CustomExponentialLR(ExponentialLR):
    """
    Add custom features for ExponentialLR
    """
    def __init__(self, optimizer, gamma, last_epoch=-1, mode="min"):
        super(CustomExponentialLR, self).__init__(optimizer,
                                                  gamma,
                                                  last_epoch=last_epoch)
        self.best = math.inf if mode == "min" else -math.inf
        self.mode = mode

    def step(self, value):
        if value < self.best and self.mode == "min":
            self.best = value
        if value > self.best and self.mode == "max":
            self.best = value
        super().step()


scheduler_cls = {
    "reduce_lr": ReduceLROnPlateau,
    "multi_step_lr": CustomMultiStepLR,
    "exponential_lr": CustomExponentialLR
}


def support_lr_scheduler(scheduler, optimizer, **kwargs):
    if scheduler not in scheduler_cls:
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")
    return scheduler_cls[scheduler](optimizer, **kwargs)