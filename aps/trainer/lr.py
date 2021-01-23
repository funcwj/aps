#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
from typing import List, Optional
from torch.optim import lr_scheduler as lr, Optimizer
from aps.libs import Register

LrScheduler = Register("lr_scheduler")


@LrScheduler.register("reduce_lr")
class ReduceLROnPlateau(lr.ReduceLROnPlateau):
    """
    Wrapper for lr.ReduceLROnPlateau
    """

    def __init__(self, *args, **kwargs):
        super(ReduceLROnPlateau, self).__init__(*args, **kwargs)


@LrScheduler.register("step_lr")
class StepLR(lr.StepLR):
    """
    Wrapper for lr.StepLR
    """

    def __init__(self, *args, **kwargs):
        super(StepLR, self).__init__(*args, **kwargs)


@LrScheduler.register("multi_step_lr")
class MultiStepLR(lr.MultiStepLR):
    """
    Wrapper for lr.MultiStepLR
    """

    def __init__(self, *args, **kwargs):
        super(MultiStepLR, self).__init__(*args, **kwargs)


@LrScheduler.register("warmup_noam_lr")
class NoamLR(lr._LRScheduler):
    """
    Lr schuduler for Transformer

    const = factor * transformer_dim^(-0.5)
    1) cur_step > warmup:   const * cur_step^(-0.5)
    2) cur_step < warmup:   const * cur_step/warmup * warmup^(-0.5)
    3) cur_step = warmup:   const * warmup^(-0.5)

    The peak value of the learning rate is
        peak_lr = factor * (transformer_dim * warmup)^(-0.5)

    Args:
        optimizer: optimizer object in torch.optim.Optimizer
        transformer_dim: transformer's model dimension
        warmup: warmup steps
        peak_lr: user defined peak learning rate if > 0
    """

    def __init__(self,
                 optimizer: Optimizer,
                 transformer_dim: int = 512,
                 peak_lr: float = -1,
                 warmup: int = 8000,
                 last_epoch: int = -1) -> None:
        self.warmup = warmup
        self.const = transformer_dim**(
            -0.5) if peak_lr <= 0 else peak_lr * warmup**0.5
        super(NoamLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self, step: Optional[int] = None) -> List[float]:
        if step is None:
            step = self._step_count
        return [
            self.const * min(step**(-0.5), step * self.warmup**(-1.5))
            for _ in self.optimizer.param_groups
        ]


@LrScheduler.register("warmup_exp_decay_lr")
class ExponentialLR(lr._LRScheduler):
    """
    Exponential scheduler proposed in SpecAugment paper：

    1) 0 < cur_step <= sr: ramp up (use sr == 0 can skip this stage)
    2) sr < cur_step <= si: hold on (use si == sr can skip this stage)
    3) si < cur_step <= sf: exponential decay
    4) cur_step > sf: hold on

    Args:
        optimizer: optimizer object in torch.optim.Optimizer
        time_stamps: [sr, si, sf] in the paper
        peak_lr: the peak value of the learning rate
        stop_lr: the minimum value of the learning rate
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-5,
                 last_epoch: int = -1) -> None:
        self.peak_lr = peak_lr
        self.sr, self.si, self.sf = time_stamps
        self.gamma = math.log(stop_lr / peak_lr) / (self.sf - self.si)
        super(ExponentialLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self, step: Optional[int] = None) -> List[float]:
        if step is None:
            step = self._step_count
        if step <= self.si:
            cur_lr = min(self.sr, step) * 1.0 * self.peak_lr / self.sr
        else:
            cur_lr = self.peak_lr * math.exp(self.gamma *
                                             (min(step, self.sf) - self.si))
        return [cur_lr for _ in self.optimizer.param_groups]


@LrScheduler.register("warmup_linear_decay_lr")
class LinearLR(lr._LRScheduler):
    """
    Linear warmup scheduler (using linear decay in ExponentialLR):

    1) 0 < cur_step <= sr: ramp up (use sr == 0 can skip this stage)
    2) sr < cur_step <= si: hold on (use si == sr can skip this stage)
    3) si < cur_step <= sf: linear decay
    4) cur_step > sf: hold on

    Args:
        optimizer: optimizer object in torch.optim.Optimizer
        time_stamps: similar with ExponentialLR
        peak_lr: the peak value of the learning rate
        stop_lr: the minimum value of the learning rate
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-8,
                 last_epoch: int = -1) -> None:
        self.peak_lr = peak_lr
        self.sr, self.si, self.sf = time_stamps
        self.gamma = (stop_lr - peak_lr) / (self.sf - self.si)
        super(LinearLR, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self, step: Optional[int] = None) -> List[float]:
        if step is None:
            step = self._step_count
        if step <= self.si:
            cur_lr = min(self.sr, step) * 1.0 * self.peak_lr / self.sr
        else:
            cur_lr = self.peak_lr + (self.gamma *
                                     (min(step, self.sf) - self.si))
        return [cur_lr for _ in self.optimizer.param_groups]
