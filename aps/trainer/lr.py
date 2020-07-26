#!/usr/bin/env python

# wujian@2020

from torch.optim import lr_scheduler


class CustomStepLR(lr_scheduler.StepLR):
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


class CustomMultiStepLR(lr_scheduler.MultiStepLR):
    """
    Using lr_conf to set milestones & gamma, e.g., 0.1@10,20,40
    """
    def __init__(self, optimizer, lr_conf="0.5@10,20,30,40", last_epoch=-1):
        gamma, milestones = self._parse_args(lr_conf)
        super(CustomMultiStepLR, self).__init__(optimizer,
                                                milestones,
                                                gamma=gamma,
                                                last_epoch=last_epoch)

    def _parse_args(self, lr_conf):
        lr_str = lr_conf.split("@")
        if len(lr_str) != 2:
            raise RuntimeError(f"Wrong format for lr_conf={lr_conf}")
        gamma = float(lr_str[0])
        milestones = list(map(int, lr_str[0].split(",")))
        return gamma, milestones

    def step(self, value):
        super().step()


class CustomExponentialLR(lr_scheduler.ExponentialLR):
    """
    To add custom features for ExponentialLR
    """
    def __init__(self, optimizer, gamma, last_epoch=-1):
        super(CustomExponentialLR, self).__init__(optimizer,
                                                  gamma,
                                                  last_epoch=last_epoch)

    def step(self, value):
        super().step()


class NoamOpt(lr_scheduler._LRScheduler):
    """
    Lr schuduler for Transformer
    """
    def __init__(self,
                 optimizer,
                 transformer_dim=512,
                 factor=1,
                 warmup=8000,
                 last_epoch=-1):
        super(NoamOpt, self).__init__(optimizer, last_epoch=last_epoch)
        self.cur_step = 1
        self.warmup = warmup
        self.const = factor * transformer_dim**(-0.5)

    def info(self):
        beg_lr = self.get_lr(1)[0]
        top_lr = self.get_lr(self.warmup)[0]
        return f"learning rate at step 1: {beg_lr:.3e} and " + \
            f"step {self.warmup:d}: {top_lr:.3e}"

    def get_lr(self, step=None):
        """
        const = factor * transformer_dim^{-0.5}
        1) cur_step > warmup:   const * cur_step**(-0.5)
        2) cur_step < warmup:   const * cur_step * warmup**(-1.5)
        3) cur_step = warmup:   const * warmup**(-0.5)
        """
        if step is None:
            step = self.cur_step
        return [
            self.const * min(step**(-0.5), step * self.warmup**(-1.5))
            for _ in self.optimizer.param_groups
        ]

    def step(self, value):
        self.cur_step += 1
        super().step()


scheduler_cls = {
    "reduce_lr": lr_scheduler.ReduceLROnPlateau,
    "step_lr": CustomStepLR,
    "multi_step_lr": CustomMultiStepLR,
    "exponential_lr": CustomExponentialLR,
    "noam_opt": NoamOpt
}


def support_lr_scheduler(scheduler, optimizer, **kwargs):
    if scheduler not in scheduler_cls:
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")
    return scheduler_cls[scheduler](optimizer, **kwargs)