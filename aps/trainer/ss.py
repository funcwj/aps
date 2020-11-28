# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Schedule sampling & Learning rate
"""

from aps.libs import Register

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
    """

    def __init__(self, ssr: float = 0) -> None:
        super(ConstScheduler, self).__init__(ssr)

    def step(self, epoch: int, accu: float) -> float:
        return self.ssr


@SsScheduler.register("trigger")
class TriggerScheduler(BaseScheduler):
    """
    Use schedule sampling rate when metrics triggered
    """

    def __init__(self, ssr: float = 0, trigger: float = 0.6) -> None:
        super(TriggerScheduler, self).__init__(ssr)
        self.trigger = trigger

    def step(self, epoch: int, accu: float) -> float:
        return 0 if accu < self.trigger else self.ssr


@SsScheduler.register("linear")
class LinearScheduler(BaseScheduler):
    """
    Use linear schedule sampling rate
    """

    def __init__(self,
                 ssr: float = 0,
                 epoch_beg: int = 10,
                 epoch_end: int = 20,
                 update_interval: int = 1) -> None:
        super(LinearScheduler, self).__init__(ssr)
        self.beg = epoch_beg
        self.end = epoch_end
        self.inc = ssr * update_interval / (epoch_end - epoch_beg)
        self.interval = update_interval

    def step(self, epoch: int, accu: float) -> float:
        if epoch < self.beg:
            return 0
        elif epoch >= self.end:
            return self.ssr
        else:
            inv = (epoch - self.beg) // self.interval + 1
            return inv * self.inc
