# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
from pathlib import Path

import torch as th
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, List, Union, NoReturn

from aps.trainer.base import Trainer
from aps.libs import ApsRegisters
import aps.distributed as dist


@ApsRegisters.trainer.register("hvd")
class HvdTrainer(Trainer):
    """
    The Horovod Trainer
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
                 acmu_gradient: int = 1,
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
                 average_checkpoint: bool = False,
                 report_metrics: List[str] = ["loss"],
                 reduction_tag: str = "none",
                 stop_on_errors: int = 10,
                 **kwargs) -> None:
        super(HvdTrainer,
              self).__init__(task,
                             rank=rank,
                             device_ids=device_ids,
                             checkpoint=checkpoint,
                             optimizer=optimizer,
                             optimizer_kwargs=optimizer_kwargs,
                             lr_scheduler=lr_scheduler,
                             lr_scheduler_kwargs=lr_scheduler_kwargs,
                             lr_scheduler_period=lr_scheduler_period,
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
                             average_checkpoint=average_checkpoint,
                             report_metrics=report_metrics,
                             stop_on_errors=stop_on_errors,
                             reduction_tag=reduction_tag)
        if dist.get_backend() != "horovod":
            raise ValueError(
                "HvdTrainer should use horovod as distributed backend")
        if not dist.hvd_available:
            raise ValueError("horovod is not installed in current machine")
        self.setup_distributed()

    def setup_distributed(self) -> NoReturn:
        """
        Setup environment for distributed training
        """
        import horovod.torch as hvd
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer, named_parameters=self.task.named_parameters())
        hvd.broadcast_parameters(self.task.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.reporter.log(f"Horovod: using horovod, rank = {self.rank}, " +
                          f"world_size={dist.world_size()}")
        self.reporter.log(
            "Horovod: BatchNorm layer will cause different dev loss on "
            "each processing due to momentum != 0 or "
            "track_running_stats = True")

    def train_one_step(self, egs: Dict) -> bool:
        """
        Make one training step for hovorod

        1) Forward & Backword
        2) Clip Gradient
        3) Step optimizer
        4) Zero optimizer
        """
        # add noise if needed
        if self.weight_noise_adder:
            self.weight_noise_adder(self.task, self.cur_step)

        is_backward_step = (self.cur_step + 1) % self.acmu_gradient == 0
        stats = self.task(egs)

        if is_backward_step:
            loss = dist.all_reduce(stats["loss"])
            loss = loss.item()
        else:
            loss = stats["loss"].item()
        # backward if not nan/inf
        if math.isfinite(loss):
            (stats["loss"] / self.acmu_gradient).backward()
        else:
            self.reporter.log(f"Invalid loss {loss:.3f}, skip...")
            return False

        # if not backward step, return
        if not is_backward_step:
            return True

        # clip gradient after backward
        norm = -1
        if self.clip_gradient:
            # for horovod
            self.optimizer.synchronize()
            norm = clip_grad_norm_(self.task.parameters(), self.clip_gradient)

        # step optimizer and update statistics
        if math.isfinite(norm):
            # for horovod
            if norm != -1:
                with self.optimizer.skip_synchronize():
                    self.optimizer.step()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
            if norm != -1:
                stats["norm"] = norm
            stats["rate"] = self.optimizer.param_groups[0]["lr"]
            self.reporter.update(egs, ["#utt", "#tok"])
            self.reporter.update(stats)
            self.lr_scheduler_step(None, end_at="step")
            return True
        else:
            self.reporter.log(f"Invalid gradient {norm:.3f}, skip...")
            return False

    def model_states(self) -> Dict:
        """
        Return model states which will be saved in the checkpoint
        """
        return {"model_state": self.task.nnet.state_dict()}
