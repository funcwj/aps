# wujian@2019

import math
from os import environ

import torch as th
import aps.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from aps.trainer.base import Trainer


class HvdTrainer(Trainer):
    """
    A Horovod Trainer
    """
    def __init__(self,
                 task,
                 rank=None,
                 device_ids=0,
                 checkpoint="cpt",
                 optimizer="adam",
                 optimizer_kwargs=None,
                 lr_scheduler="reduce_lr",
                 lr_scheduler_kwargs=None,
                 ss_scheduler="const",
                 ss_scheduler_kwargs=None,
                 clip_gradient=None,
                 gaussian_noise_std=None,
                 prog_interval=100,
                 save_interval=-1,
                 resume="",
                 init="",
                 tensorboard=False,
                 stop_criterion="loss",
                 no_impr=6,
                 no_impr_thres=1e-3):
        super(HvdTrainer,
              self).__init__(task,
                             rank=rank,
                             device_ids=device_ids,
                             checkpoint=checkpoint,
                             optimizer=optimizer,
                             optimizer_kwargs=optimizer_kwargs,
                             lr_scheduler=lr_scheduler,
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
        if dist.get_backend() != "horovod":
            raise ValueError(
                f"aps.distributed doesn't use horovod as backend")
        if not dist.hvd_available:
            raise ValueError(
                f"horovod is not installed in current environment")
        self.setup_distributed()

    def setup_distributed(self):
        """
        Setup environment for distributed training
        """
        import horovod.torch as hvd
        hvd.broadcast_parameters(self.task.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)
        self.optimizer = hvd.DistributedOptimizer(
            self.optimizer, named_parameters=self.task.named_parameters())
        self.reporter.log(f"HVD: using horovod, rank = {self.rank}, " +
                          f"world_size={dist.world_size()}")

    def save_checkpoint(self, epoch, best=True):
        """
        Save checkpoint (epoch, model, optimizer)
        """
        if self.rank in [0, None]:
            cpt = {
                "epoch": epoch,
                "model_state_dict": self.task.nnet.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_dict": self.lr_scheduler.state_dict()
            }
            cpt_name = "{}.pt.tar".format("best" if best else "last")
            th.save(cpt, self.checkpoint / cpt_name)
            self.reporter.log(f"Save checkpoint {cpt_name}")
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")