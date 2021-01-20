# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Schedule sampling
"""

from aps.libs import Register
from typing import List

SsScheduler = Register("ss_scheduler")


class BaseScheduler(object):
    """
    Basic class for schedule sampling
    """

    def __init__(self, ssr: float) -> None:
        self.ssr = ssr

    def step(self, epoch, accu):
        raise NotImplementedError


@SsScheduler.register("const")
class ConstScheduler(BaseScheduler):
    """
    Use const schedule sampling rate
    Args:
        ssr: value of schedule sampling rate
    """

    def __init__(self, ssr: float = 0) -> None:
        super(ConstScheduler, self).__init__(ssr)

    def step(self, epoch: int, accu: float) -> float:
        return self.ssr


@SsScheduler.register("epoch")
class EpochScheduler(BaseScheduler):
    """
    Do schedule sampling during several epochs
    Args:
        ssr: value of schedule sampling rate
        epoch_beg: do schedule sampling from epoch #epoch_beg
        epoch_end: stop schedule sampling at epoch #epoch_end
    """

    def __init__(self,
                 ssr: float = 0,
                 epochs: List[int] = [10, 20],
                 epoch_end: int = 20) -> None:
        super(EpochScheduler, self).__init__(ssr)
        self.beg, self.end = epochs

    def step(self, epoch: int, accu: float) -> float:
        if self.beg <= epoch and epoch <= self.end:
            return self.ssr
        return 0


@SsScheduler.register("trigger")
class TriggerScheduler(BaseScheduler):
    """
    Use schedule sampling rate when metrics triggered
    Args:
        ssr: value of schedule sampling rate
        trigger: do schedule sampling when accu > #trigger
    """

    def __init__(self, ssr: float = 0, trigger: float = 0.6) -> None:
        super(TriggerScheduler, self).__init__(ssr)
        self.trigger = trigger

    def step(self, epoch: int, accu: float) -> float:
        return 0 if accu < self.trigger else self.ssr


@SsScheduler.register("linear")
class LinearScheduler(BaseScheduler):
    """
    Use linear increasing schedule sampling rate
    Args:
        ssr: value of schedule sampling rate
        epoch_beg: do schedule sampling from epoch #epoch_beg
        epoch_end: stop schedule sampling at epoch #epoch_end
        update_interval: the interval to increase the ssr
    """

    def __init__(self,
                 ssr: float = 0,
                 epochs: List[int] = [10, 20],
                 update_interval: int = 1) -> None:
        super(LinearScheduler, self).__init__(ssr)
        self.beg, self.end = epochs
        self.inc = ssr * update_interval / (self.end - self.beg)
        self.interval = update_interval

    def step(self, epoch: int, accu: float) -> float:
        if epoch < self.beg:
            return 0
        elif epoch >= self.end:
            return self.ssr
        else:
            inv = (epoch - self.beg) // self.interval + 1
            return inv * self.inc
