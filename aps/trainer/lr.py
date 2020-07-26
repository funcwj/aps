#!/usr/bin/env python

# wujian@2020

import math

from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR, ExponentialLR


class CustomStepLR(StepLR):
    """
    To add custom features for StepLR
    """
    def __init__(self, optimizer, step_size=10, gamma=0.1, last_epoch=-1):
        super(CustomStepLR, self).__init__(optimizer,
                                           step_size,
                                           gamma=gamma,
                                           last_epoch=last_epoch)

    def step(self, value):
        super().step()


class CustomMultiStepLR(MultiStepLR):
    """
    Using lr_conf to set milestones & gamma, e.g., 0.1@10,20,40
    """
    def __init__(self, optimizer, lr_conf="0.5@10,20,30,40", last_epoch=-1):
        lr_str = lr_conf.split("@")
        if len(lr_str) != 2:
            raise RuntimeError(f"Wrong format for lr_conf={lr_conf}")
        gamma = float(lr_str[0])
        milestones = list(map(int, lr_str[0].split(",")))
        super(CustomMultiStepLR, self).__init__(optimizer,
                                                milestones,
                                                gamma=gamma,
                                                last_epoch=last_epoch)

    def step(self, value):
        super().step()


class CustomExponentialLR(ExponentialLR):
    """
    To add custom features for ExponentialLR
    """
    def __init__(self, optimizer, gamma, last_epoch=-1):
        super(CustomExponentialLR, self).__init__(optimizer,
                                                  gamma,
                                                  last_epoch=last_epoch)

    def step(self, value):
        super().step()


scheduler_cls = {
    "reduce_lr": ReduceLROnPlateau,
    "step_lr": CustomStepLR,
    "multi_step_lr": CustomMultiStepLR,
    "exponential_lr": CustomExponentialLR
}


def support_lr_scheduler(scheduler, optimizer, **kwargs):
    if scheduler not in scheduler_cls:
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")
    return scheduler_cls[scheduler](optimizer, **kwargs)