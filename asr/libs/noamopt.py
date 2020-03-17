#!/usr/bin/env python

import torch as th


class NoamOpt(object):
    """
    Optimizer for Transformer
    """
    def __init__(self,
                 parameters,
                 transformer_dim=512,
                 factor=10,
                 warmup=25000):
        self.cur_step = 1
        self.warmup = warmup
        self.factor = factor
        self.transformer_dim = transformer_dim
        self.cur_lr = self._lr()
        self.optimizer = th.optim.Adam(parameters,
                                       lr=self.cur_lr,
                                       betas=(0.9, 0.98),
                                       eps=1e-9)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def _lr(self):
        return self.factor * self.transformer_dim**(-0.5) * min(
            self.cur_step**(-0.5), self.cur_step * self.warmup**(-1.5))

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
            "warmup": self.warmup,
            "factor": self.factor,
            "transformer_dim": self.transformer_dim,
            "optimizer": self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)