# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math

import torch as th
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, List, Union, NoReturn
from pathlib import Path

from aps.trainer.base import Trainer
from aps.libs import ApsRegisters

import aps.distributed as dist


@ApsRegisters.trainer.register("ddp")
class DdpTrainer(Trainer):
    """
    A PyTorch distributed data parallel (DDP) Trainer
    """

    def __init__(self,
                 task: th.nn.Module,
                 rank: Optional[int] = None,
                 device_ids: Union[str, int, List[int]] = 0,
                 checkpoint: Union[str, Path] = "cpt",
                 optimizer: str = "adam",
                 optimizer_kwargs: Optional[Dict] = None,
                 lr_scheduler: str = "reduce_lr",
                 lr_scheduler_kwargs: Optional[Dict] = None,
                 lr_scheduler_period: str = "epoch",
                 ss_scheduler: str = "const",
                 ss_scheduler_kwargs: Optional[Dict] = None,
                 clip_gradient: Optional[float] = None,
                 acmu_gradient: int = -1,
                 weight_noise_cfg: List[int] = [0, 1, -1],
                 weight_noise_std: Optional[float] = None,
                 prog_interval: int = 100,
                 save_interval: int = -1,
                 resume: str = "",
                 init: str = "",
                 tensorboard: bool = False,
                 stop_criterion: str = "loss",
                 no_impr: int = 6,
                 no_impr_thres: float = 1e-3,
                 report_metrics: List[str] = ["loss"],
                 report_reduction: str = "none",
                 stop_on_errors: int = 10,
                 **kwargs) -> None:
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
                             acmu_gradient=acmu_gradient,
                             weight_noise_cfg=weight_noise_cfg,
                             weight_noise_std=weight_noise_std,
                             prog_interval=prog_interval,
                             save_interval=save_interval,
                             resume=resume,
                             init=init,
                             tensorboard=tensorboard,
                             stop_criterion=stop_criterion,
                             no_impr=no_impr,
                             no_impr_thres=no_impr_thres,
                             report_metrics=report_metrics,
                             report_reduction=report_reduction,
                             stop_on_errors=stop_on_errors)
        if dist.get_backend() not in ["torch", "none"]:
            raise ValueError(
                "DdpTrainer should use torch/none as distributed backend")
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

    def train_one_step(self, egs: Dict) -> bool:
        """
        Make one training step (return true if no error exists)

        1) Zero optimizer
        2) Forward & Backword
        3) Clip Gradient
        4) Step optimizer
        """
        self.optimizer.zero_grad()

        # add noise if needed
        if self.weight_noise_adder:
            self.weight_noise_adder(self.task, self.cur_step)

        stats = self.task(egs)
        # use all reduce to check loss
        if self.distributed:
            loss = dist.all_reduce(stats["loss"].clone()).item()
        else:
            loss = stats["loss"].item()

        # backward if not nan/inf
        if math.isfinite(loss):
            stats["loss"].backward()
        else:
            self.reporter.log(f"Invalid loss {loss:.3f}, skip...")
            return False

        # clip gradient after backward
        norm = -1
        if self.clip_gradient:
            norm = clip_grad_norm_(self.task.parameters(), self.clip_gradient)

        # step optimizer and update statistics
        if math.isfinite(norm):
            self.optimizer.step()
            if norm != -1:
                stats["norm"] = norm
            stats["rate"] = self.optimizer.param_groups[0]["lr"]
            self.reporter.update(egs, ["#utt", "#tok"])
            self.reporter.update(stats)
            # schedule lr if needed
            self.lr_scheduler_step(None, end_at="step")
            return True
        else:
            self.reporter.log(f"Invalid gradient {norm:.3f}, skip...")
            return False

    def model_states(self) -> Dict:
        """
        Return model states which will be saved in the checkpoint
        """
        return {
            "model_state":
                self.task.module.nnet.state_dict()
                if self.distributed else self.task.nnet.state_dict()
        }
