#!/usr/bin/env python

# wujian@2020

from torch.optim import lr_scheduler


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


class NoamLR(lr_scheduler._LRScheduler):
    """
    Lr schuduler for Transformer
    """

    def __init__(self,
                 optimizer,
                 transformer_dim=512,
                 factor=1,
                 warmup=8000,
                 last_epoch=-1):
        self.warmup = warmup
        self.const = factor * transformer_dim**(-0.5)
        super(NoamLR, self).__init__(optimizer, last_epoch=last_epoch)

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
            step = self._step_count
        return [
            self.const * min(step**(-0.5), step * self.warmup**(-1.5))
            for _ in self.optimizer.param_groups
        ]


scheduler_cls = {
    "reduce_lr": lr_scheduler.ReduceLROnPlateau,
    "step_lr": lr_scheduler.StepLR,
    "multi_step_lr": CustomMultiStepLR,
    "exponential_lr": lr_scheduler.ExponentialLR,
    "noam_lr": NoamLR
}


def support_lr_scheduler(scheduler, optimizer, **kwargs):
    if scheduler not in scheduler_cls:
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")
    return scheduler_cls[scheduler](optimizer, **kwargs)
