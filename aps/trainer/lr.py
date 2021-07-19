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


class WarmupDecayBase(lr._LRScheduler):
    """
    Base class for warmup and decay lr scheduler:

    1) 0 < cur_step <= warmup: ramp up (use warmup == 0 can skip this stage)
    2) warmup < cur_step <= holdon: hold on (use warmup == holdon can skip this stage)
    3) holdon < cur_step <= max_steps: user-defined decay policy
    4) cur_step > max_steps: hold on

    Args:
        optimizer: optimizer object in torch.optim.Optimizer
        time_stamps: [warmup steps, holdon steps, max steps]
        peak_lr: the peak value of the learning rate
        stop_lr: the minimum value of the learning rate
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-5,
                 last_epoch: int = -1) -> None:
        self.gamma = None
        self.peak_lr, self.stop_lr = peak_lr, stop_lr
        self.warmup, self.holdon, self.max_steps = time_stamps
        super(WarmupDecayBase, self).__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self, step: Optional[int] = None) -> List[float]:
        if step is None:
            step = self._step_count
        if step <= self.holdon:
            cur_lr = min(self.warmup, step) * 1.0 * self.peak_lr / self.warmup
        elif step >= self.max_steps:
            cur_lr = self.stop_lr
        else:
            cur_lr = self._decay_lr(step)
        return [cur_lr for _ in self.optimizer.param_groups]

    def _decay_lr(self, step: int):
        raise NotImplementedError()


@LrScheduler.register("warmup_exp_decay_lr")
class ExponentialDecayLR(WarmupDecayBase):
    """
    Warmup and exponential decay lr scheduler proposed in SpecAugment paper
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-5,
                 last_epoch: int = -1) -> None:
        super(ExponentialDecayLR, self).__init__(optimizer,
                                                 time_stamps=time_stamps,
                                                 peak_lr=peak_lr,
                                                 stop_lr=stop_lr,
                                                 last_epoch=last_epoch)

    def _decay_lr(self, step: int):
        if self.gamma is None:
            self.gamma = math.log(
                self.stop_lr / self.peak_lr) / (self.max_steps - self.holdon)
        return self.peak_lr * math.exp(self.gamma * (step - self.holdon))


@LrScheduler.register("warmup_linear_decay_lr")
class LinearDecayLR(WarmupDecayBase):
    """
    Warmup and linear decay lr scheduler
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-8,
                 last_epoch: int = -1) -> None:
        super(LinearDecayLR, self).__init__(optimizer,
                                            time_stamps=time_stamps,
                                            peak_lr=peak_lr,
                                            stop_lr=stop_lr,
                                            last_epoch=last_epoch)

    def _decay_lr(self, step: int):
        if self.gamma is None:
            self.gamma = (self.stop_lr - self.peak_lr) / (self.max_steps -
                                                          self.holdon)
        return self.peak_lr + (self.gamma * (step - self.holdon))


@LrScheduler.register("warmup_cos_decay_lr")
class CosineDecayLR(WarmupDecayBase):
    """
    Warmup and cosine decay lr scheduler
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-8,
                 last_epoch: int = -1) -> None:
        super(CosineDecayLR, self).__init__(optimizer,
                                            time_stamps=time_stamps,
                                            peak_lr=peak_lr,
                                            stop_lr=stop_lr,
                                            last_epoch=last_epoch)

    def _decay_lr(self, step: int):
        if self.gamma is None:
            self.gamma = math.pi / (self.max_steps - self.holdon)
        return (self.peak_lr - self.stop_lr) * (
            1 + math.cos(self.gamma * (step - self.holdon))) / 2 + self.stop_lr


@LrScheduler.register("warmup_power_decay_lr")
class PowerDecayLR(WarmupDecayBase):
    """
    Warmup and power (e.g., square or sqrt) decay lr scheduler
    """

    def __init__(self,
                 optimizer: Optimizer,
                 time_stamps: List[int] = [1000, 4000, 16000],
                 power: float = 2,
                 peak_lr: float = 1e-3,
                 stop_lr: float = 1e-8,
                 last_epoch: int = -1) -> None:
        self.power = power
        super(PowerDecayLR, self).__init__(optimizer,
                                           time_stamps=time_stamps,
                                           peak_lr=peak_lr,
                                           stop_lr=stop_lr,
                                           last_epoch=last_epoch)

    def _decay_lr(self, step: int):
        if self.gamma is None:
            self.gamma = 1 / (self.max_steps - self.holdon)
        return (self.peak_lr - self.stop_lr) * (
            (self.max_steps - step) * self.gamma)**self.power + self.stop_lr
