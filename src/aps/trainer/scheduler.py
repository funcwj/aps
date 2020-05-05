"""
Schedule sampling & Learning rate
"""

import torch as th
from torch.optim import lr_scheduler


def support_ss_scheduler(scheduler, **kwargs):
    """
    Return supported ss scheduler
    """
    scheduler_templ = {
        "const": ConstScheduler,
        "linear": LinearScheduler,
        "trigger": TriggerScheduler
    }
    if scheduler not in scheduler_templ:
        raise RuntimeError(f"Not supported scheduler: {scheduler}")
    return scheduler_templ[scheduler](**kwargs)


class SsScheduler(object):
    """
    Basic class for schedule sampling
    """
    def __init__(self, ssr):
        self.ssr = ssr

    def step(self, epoch, accu):
        raise NotImplementedError


class ConstScheduler(SsScheduler):
    """
    Use const schedule sampling rate
    """
    def __init__(self, ssr=0):
        super(ConstScheduler, self).__init__(ssr)

    def step(self, epoch, accu):
        return self.ssr


class TriggerScheduler(SsScheduler):
    """
    Use schedule sampling rate when metrics triggered
    """
    def __init__(self, ssr=0, trigger=0.6):
        super(TriggerScheduler, self).__init__(ssr)
        self.trigger = trigger

    def step(self, epoch, accu):
        return 0 if accu < self.trigger else self.ssr


class LinearScheduler(SsScheduler):
    """
    Use linear schedule sampling rate
    """
    def __init__(self, ssr=0, epoch_beg=10, epoch_end=20, update_interval=1):
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


class NoamOpt(object):
    """
    Optimizer for Transformer
    """
    def __init__(self, parameters, transformer_dim=512, factor=1, warmup=8000):
        self.cur_step = 1
        self.warmup = warmup
        self.const = factor * transformer_dim**(-0.5)
        self.cur_lr = self._lr()
        self.optimizer = th.optim.Adam(parameters,
                                       lr=self.cur_lr,
                                       betas=(0.9, 0.98),
                                       eps=1e-9)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def info(self):
        beg_lr = self._lr(1)
        top_lr = self._lr(self.warmup)
        return f"learning rate at step 1: {beg_lr:.3e} and " + \
            f"step {self.warmup:d}: {top_lr:.3e}"

    def _lr(self, step=None):
        """
        const = factor * transformer_dim^{-0.5}
        1) cur_step > warmup:   const * cur_step**(-0.5)
        2) cur_step < warmup:   const * cur_step * warmup**(-1.5)
        3) cur_step = warmup:   const * warmup**(-0.5)
        """
        if step is None:
            step = self.cur_step
        return self.const * min(step**(-0.5), step * self.warmup**(-1.5))

    def step(self):
        self.cur_lr = self._lr()
        for param in self.optimizer.param_groups:
            param["lr"] = self.cur_lr
        self.optimizer.step()
        self.cur_step += 1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return {
            "cur_step": self.cur_step,
            "cur_lr": self.cur_lr,
            "const": self.const,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)