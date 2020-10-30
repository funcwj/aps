# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
from os import environ
from pathlib import Path

import torch as th
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Union, Tuple, NoReturn

from aps.trainer.base import Trainer
import aps.distributed as dist


class DdpTrainer(Trainer):
    """
    A PyTorch distributed data parallel (DDP) Trainer
    """

    def __init__(self,
                 task: th.nn.Module,
                 rank: Optional[int] = None,
                 device_ids: int = 0,
                 checkpoint: Union[str, Path] = "cpt",
                 optimizer: str = "adam",
                 optimizer_kwargs: Optional[Dict] = None,
                 lr_scheduler: str = "reduce_lr",
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 lr_scheduler_period: str = "epoch",
                 ss_scheduler: str = "const",
                 ss_scheduler_kwargs: Optional[Dict] = None,
                 clip_gradient: Optional[int] = None,
                 gaussian_noise_std: Optional[float] = None,
                 prog_interval: int = 100,
                 save_interval: int = -1,
                 resume: int = "",
                 init: int = "",
                 tensorboard: bool = False,
                 stop_criterion: str = "loss",
                 no_impr: int = 6,
                 no_impr_thres: float = 1e-3) -> None:
        super(DdpTrainer,
              self).__init__(task,
                             rank=rank,
                             device_ids=device_ids,
                             checkpoint=checkpoint,
                             optimizer=optimizer,
                             optimizer_kwargs=optimizer_kwargs,
                             lr_scheduler=lr_scheduler,
                             lr_scheduler_period=lr_scheduler_period,
                             lr_scheduler_kwargs=lr_scheduler_kwargs,
                             ss_scheduler=ss_scheduler,
                             ss_scheduler_kwargs=ss_scheduler_kwargs,
                             clip_gradient=clip_gradient,
                             gaussian_noise_std=gaussian_noise_std,
                             prog_interval=prog_interval,
                             save_interval=save_interval,
                             resume=resume,
                             init=init,
                             tensorboard=tensorboard,
                             stop_criterion=stop_criterion,
                             no_impr=no_impr,
                             no_impr_thres=no_impr_thres)
        if dist.get_backend() not in ["torch", "none"]:
            raise ValueError(f"aps.distributed doesn't use torch as backend")
        self.setup_distributed()

    def setup_distributed(self) -> NoReturn:
        """
        Setup environment for distributed training
        """
        if self.cuda_devices >= 2:
            self.distributed = True
            self.reporter.log(
                f"DDP: using distributed data parallel (DDP), rank={self.rank}, "
                + f"world_size={dist.world_size()}")
            self.task = DistributedDataParallel(self.task,
                                                device_ids=[self.rank])
        else:
            self.distributed = False

    def save_checkpoint(self, epoch: int, best: bool = True) -> NoReturn:
        """
        Save checkpoint (epoch, model, optimizer)
        """
        if self.rank in [0, None]:
            cpt = {
                "epoch":
                    epoch,
                "model_state_dict":
                    self.task.module.nnet.state_dict()
                    if self.distributed else self.task.nnet.state_dict(),
                "optim_state_dict":
                    self.optimizer.state_dict(),
                "lr_scheduler_dict":
                    self.lr_scheduler.state_dict()
            }
            cpt_name = "{}.pt.tar".format("best" if best else "last")
            th.save(cpt, self.checkpoint / cpt_name)
            self.reporter.log(f"Save checkpoint {self.checkpoint / cpt_name}")
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")
