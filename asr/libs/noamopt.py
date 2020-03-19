#!/usr/bin/env python

import torch as th


class NoamOpt(object):
    """
    Optimizer for Transformer
    """
    def __init__(self,
                 parameters,
                 transformer_dim=512,
                 factor=1,
                 warmup=8000):
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