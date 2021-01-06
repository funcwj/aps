# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

try:
    import apex
    apex_available = True
except ImportError:
    apex_available = False

import math
import torch as th

from pathlib import Path
from typing import Optional, Dict, List, Union, NoReturn

from torch.nn.utils import clip_grad_norm_
from aps.trainer.ddp import Trainer
from aps.libs import ApsRegisters
import aps.distributed as dist


@ApsRegisters.trainer.register("apex")
class ApexTrainer(Trainer):
    """
    Trainer using the NVIDIA's apex (https://github.com/NVIDIA/apex)
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
                 weight_noise_cfg: List[int] = [0, 1, -1],
                 weight_noise_std: Optional[float] = None,
                 prog_interval: int = 100,
                 save_interval: int = -1,
                 resume: str = "",
                 init: str = "",
                 tensorboard: bool = False,
                 stop_criterion: str = "loss",
                 opt_level: str = "O0",
                 no_impr: int = 6,
                 no_impr_thres: float = 1e-3,
                 report_metrics: List[str] = ["loss"],
                 reduction_tag: str = "none",
                 stop_on_errors: int = 10,
                 **kwargs) -> None:
        super(ApexTrainer,
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
                             stop_on_errors=stop_on_errors,
                             reduction_tag=reduction_tag)
        if dist.get_backend() not in ["torch", "none"]:
            raise ValueError(
                "ApexTrainer should use torch/none as distributed backend")
        if not apex_available:
            raise ValueError("apex is not installed in current machine")
        self.setup_distributed(opt_level)

    def setup_distributed(self, opt_level: str) -> NoReturn:
        """
        Setup environment for apex distributed training
        """
        # using apex synced BN
        self.task = apex.parallel.convert_syncbn_model(self.task)
        # O0: FP32 training & O3 FP16 training
        self.task, self.optimizer = apex.amp.initialize(self.task,
                                                        self.optimizer,
                                                        opt_level=opt_level)
        self.reporter.log(f"Apex: Using opt-level {opt_level}")
        if self.cuda_devices >= 2:
            self.distributed = True
            self.reporter.log(
                f"Apex: using distributed data parallel (DDP), rank={self.rank}, "
                + f"world_size={dist.world_size()}")
            self.task = apex.parallel.DistributedDataParallel(
                self.task, delay_allreduce=True)
        else:
            self.distributed = False
        # restore amp stats
        if self.cpt_stats:
            apex.amp.load_state_dict(self.cpt_stats["amp_state"])

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

        stats = self.task(egs)
        # use all reduce to check loss
        if self.distributed:
            loss = dist.all_reduce(stats["loss"].clone()).item()
        else:
            loss = stats["loss"].item()
        # backward if not nan/inf
        if math.isfinite(loss):
            with apex.amp.scale_loss(stats["loss"],
                                     self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.reporter.log(f"Invalid loss {loss:.3f}, skip...")
            return False

        # clip gradient after backward
        norm = -1
        if self.clip_gradient:
            # for apex
            norm = clip_grad_norm_(apex.amp.master_params(self.optimizer),
                                   self.clip_gradient)

        # step optimizer and update statistics
        if math.isfinite(norm):
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
        return {
            "amp_state":
                apex.amp.state_dict(),
            "model_state":
                self.task.module.nnet.state_dict()
                if self.distributed else self.task.nnet.state_dict()
        }
