"""
Schedule sampling
"""


def support_ss_scheduler(scheduler, prob, **kwargs):
    scheduler_templ = {
        "const": ConstScheduler,
        "linear": LinearScheduler,
        "trigger": TriggerScheduler
    }
    if scheduler not in scheduler_templ:
        raise RuntimeError(f"Not supported scheduler: {scheduler}")
    return scheduler_templ[scheduler](prob, **kwargs)


class Scheduler(object):
    """
    Basic class for schedule sampling
    """
    def __init__(self, ssr, **kwargs):
        self.ssr = ssr

    def step(self, epoch, accu):
        raise NotImplementedError


class ConstScheduler(Scheduler):
    """
    Use const schedule sampling rate
    """
    def __init__(self, ssr):
        super(ConstScheduler, self).__init__(ssr)

    def step(self, epoch, accu):
        return self.ssr


class TriggerScheduler(Scheduler):
    """
    Use schedule sampling rate when metrics triggered
    """
    def __init__(self, ssr, trigger=0.6):
        super(TriggerScheduler, self).__init__(ssr)
        self.trigger = trigger

    def step(self, epoch, accu):
        return 0 if accu < self.trigger else self.ssr


class LinearScheduler(Scheduler):
    """
    Use linear schedule sampling rate
    """
    def __init__(self, ssr, epoch_beg=10, epoch_end=20, update_interval=1):
        super(LinearScheduler, self).__init__(ssr)
        self.beg = epoch_beg
        self.end = epoch_end
        self.inc = ssr * update_interval / (epoch_end - epoch_beg)
        self.interval = update_interval

    def step(self, epoch, accu):
        if epoch < self.beg:
            return 0
        elif epoch >= self.end:
            return self.ssr
        else:
            inv = (epoch - self.beg) // self.interval + 1
            return inv * self.inc