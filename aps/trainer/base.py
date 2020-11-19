# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import warnings

from pathlib import Path
from collections import defaultdict

import torch as th
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Union, Tuple, NoReturn, Iterable
from aps.trainer.ss import ss_scheduler_cls
from aps.trainer.lr import lr_scheduler_cls

from aps.utils import load_obj, get_device_ids, get_logger, SimpleTimer
from aps.task import Task


def add_gaussian_noise(nnet: th.nn.Module, std: float = 0.075) -> NoReturn:
    """
    Add gaussian noise to updated weights
    """
    for p in nnet.parameters():
        if p.requires_grad:
            p.data += th.randn(p.data.shape, device=nnet.device) * std


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self,
                 checkpoint: Path,
                 period: int = 100,
                 tensorboard: bool = True,
                 rank: Optional[int] = None) -> None:
        self.period = period
        self.rank = rank
        if rank is None:
            logger_loc = (checkpoint / "trainer.log").as_posix()
            self.header = "Trainer"
        else:
            logger_loc = (checkpoint / f"trainer.rank.{rank}.log").as_posix()
            self.header = f"Rank {rank}"

        self.logger = get_logger(logger_loc, file=True)
        # only for rank-0
        if tensorboard and rank in [0, None]:
            self.board_writer = SummaryWriter(checkpoint)
        else:
            self.board_writer = None
        self.reset()

    def log(self, sstr: str) -> NoReturn:
        self.logger.info(f"{self.header}: {sstr}")

    def eval(self) -> NoReturn:
        self.log(">> Set eval mode ...")
        self.mode = "valid"
        self.reset()

    def train(self) -> NoReturn:
        self.log(">> Set train mode ...")
        self.mode = "train"
        self.reset()

    def reset(self) -> NoReturn:
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def update(self, dict_obj: Dict) -> NoReturn:
        if dict_obj is None:
            return
        for key, value in dict_obj.items():
            if isinstance(value, th.Tensor):
                value = value.item()
            self.add(key, value)

    def add(self, key: str, value: float) -> NoReturn:
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            if key == "rate":
                cur = self.stats[key][-1]
                self.log(f"Processed {N:.2e} batches ({key} = {cur:.3e}) ...")
            else:
                avg = sum(self.stats[key][-self.period:]) / self.period
                self.log(f"Processed {N:.2e} batches ({key} = {avg:+.2f}) ...")

    def report(self, epoch: int, lr: float) -> Tuple[float, float, str]:
        N = len(self.stats["loss"])
        if self.mode == "valid":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"Loss on {N:d} batches: {sstr}")

        if N == 0:
            raise RuntimeError("No statistics to report")
        loss = sum(self.stats["loss"]) / N
        accu = sum(
            self.stats["accu"]) * 100 / N if "accu" in self.stats else None
        if self.board_writer:
            self.board_writer.add_scalar(f"loss/{self.mode}", loss, epoch)
            if accu is not None:
                self.board_writer.add_scalar(f"accu/{self.mode}", accu, epoch)
        cost = self.timer.elapsed()
        if accu is not None:
            hstr = f"Loss/Accu(time/#batch, lr={lr:.3e}) - Epoch {epoch:2d}: "
            cstr = f"{self.mode} = {loss:.4f}/{accu:.2f}({cost:.2f}m/{N:d})"
        else:
            hstr = f"Loss(time/#batch, lr={lr:.3e}) - Epoch {epoch:2d}: "
            cstr = f"{self.mode} = {loss:.4f}({cost:.2f}m/{N:d})"
        return loss, accu, hstr + cstr


class StopCriterion(object):
    """
    Early stop of the training
    """

    def __init__(self,
                 no_impr: int,
                 mode: str = "min",
                 init_criterion: float = math.inf,
                 no_impr_thres: float = 2e-3) -> None:
        self.max_no_impr = no_impr
        self.no_impr = 0
        self.no_impr_thres = no_impr_thres
        self.mode = mode
        self.best_criterion = init_criterion

    def reset(self, update_value: float) -> NoReturn:
        self.best_criterion = update_value

    def stop(self) -> bool:
        return self.no_impr == self.max_no_impr

    @property
    def best(self) -> float:
        return self.best_criterion

    def step(self, update_value: float) -> bool:
        is_better = True
        # loss
        if self.mode == "min":
            is_better = self.best_criterion > update_value + self.no_impr_thres
        # accu
        if self.mode == "max":
            is_better = self.best_criterion < update_value - self.no_impr_thres
        if is_better:
            self.best_criterion = update_value
            self.no_impr = 0
            return True
        else:
            self.no_impr += 1
            return False


class Trainer(object):
    """
    A PyTorch distributed trainer
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
                 gaussian_noise_std: Optional[float] = None,
                 prog_interval: int = 100,
                 save_interval: int = -1,
                 resume: str = "",
                 init: str = "",
                 tensorboard: bool = False,
                 stop_criterion: str = "loss",
                 no_impr: int = 6,
                 no_impr_thres: float = 1e-3,
                 **kwargs) -> None:
        if not isinstance(task, Task):
            raise TypeError(
                f"Trainer accepts Task object, but got {type(task)}")
        if lr_scheduler_period not in ["epoch", "step"]:
            raise TypeError(
                f"Unsupported lr_scheduler_period: {lr_scheduler_period}")
        if rank is not None and rank < 0:
            raise ValueError(f"Got invalid rank value: {rank}")
        if not isinstance(device_ids, tuple):
            device_ids = get_device_ids(device_ids)
        self.cuda_devices = len(device_ids)
        self.device_ids = device_ids

        if rank is None:
            # single GPU
            self.default_device = th.device(f"cuda:{device_ids[0]:d}")
        else:
            # in distributed mode
            if rank >= self.cuda_devices:
                raise ValueError("rank value exceeds number of GPUs: " +
                                 f"{rank} vs {self.cuda_devices}")
            self.default_device = th.device(f"cuda:{device_ids[rank]:d}")

        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)

        self.rank = rank
        self.checkpoint = Path(checkpoint)
        # if exist, resume training
        last_checkpoint = self.checkpoint / "last.pt.tar"
        if last_checkpoint.exists():
            resume = last_checkpoint.as_posix()

        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self.reporter = ProgressReporter(self.checkpoint,
                                         rank=rank,
                                         period=prog_interval,
                                         tensorboard=tensorboard)

        self.clip_gradient = clip_gradient
        self.gaussian_noise_std = gaussian_noise_std
        self.cur_epoch = 0  # zero based
        self.save_interval = save_interval
        self.ssr = 0
        self.no_impr = no_impr

        mode = "max" if stop_criterion == "accu" else "min"
        self.stop_on = stop_criterion
        self.stop_criterion = StopCriterion(no_impr,
                                            mode=mode,
                                            no_impr_thres=no_impr_thres)

        self.num_params = sum(
            [param.nelement() for param in task.nnet.parameters()]) / 10.0**6
        self.task = task
        if self.rank in [0, None]:
            self.reporter.log(f"Model summary:\n{task.nnet}")
        self.task.to(self.default_device)

        if resume or init:
            self.cpt_stats, optimizer_dict = self.load_checkpoint(
                resume if resume else init, "resume" if resume else "init")
            lr_scheduler_kwargs["state"] = self.cpt_stats["lr_scheduler_dict"]
        else:
            self.cpt_stats, optimizer_dict = None, None
            lr_scheduler_kwargs["state"] = None
        # make optimizer
        self.optimizer = self.create_optimizer(optimizer,
                                               optimizer_kwargs,
                                               state=optimizer_dict)

        # make lr scheduler
        if lr_scheduler == "reduce_lr":
            if lr_scheduler_period != "epoch":
                warnings.warn("For reduce_lr scheduler, lr_scheduler_period " +
                              "shoule be \'epoch\'")
                lr_scheduler_period = "epoch"
                reduce_lr_kwargs = {
                    "mode": mode,
                    "threshold_mode": "abs",
                    "threshold": no_impr_thres
                }
                lr_scheduler_kwargs.update(reduce_lr_kwargs)
        self.lr_scheduler = self.create_scheduler(lr_scheduler, self.optimizer,
                                                  **lr_scheduler_kwargs)
        self.lr_scheduler_period = lr_scheduler_period

        # make ss scheduler
        if ss_scheduler_kwargs:
            if ss_scheduler not in ss_scheduler_cls:
                raise ValueError(f"Unsupported ss scheduler: {ss_scheduler}")
            self.ss_scheduler = ss_scheduler_cls[ss_scheduler](
                **ss_scheduler_kwargs)
            self.reporter.log(f"Using schedule sampling: {ss_scheduler}")
        else:
            self.ss_scheduler = None

        # logging
        if rank is None:
            self.reporter.log(f"Loading model to GPU:{device_ids[0]}, " +
                              f"#param: {self.num_params:.2f}M")
        else:
            self.reporter.log(
                f"Loading model to GPU-{rank}/{self.cuda_devices}, " +
                f"#param: {self.num_params:.2f}M")

        self.reporter.log(f"Stop criterion: {self.stop_on}")
        if clip_gradient:
            self.reporter.log(
                f"Gradient clipping if over {clip_gradient} L2 norm")
        if gaussian_noise_std:
            self.reporter.log("Add gaussian noise to weights, with " +
                              f"std = {gaussian_noise_std}")

    def create_optimizer(self,
                         optimizer: str,
                         kwargs: Dict,
                         state: Optional[Dict] = None) -> th.optim.Optimizer:
        """
        Return a PyTorch optimizer
        """
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax,  # lr, weight_decay
            "adamw": th.optim.AdamW,  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        opt = supported_optimizer[optimizer](self.task.parameters(), **kwargs)
        self.reporter.log(f"Create optimizer {optimizer}: {kwargs}")
        if state is not None:
            opt.load_state_dict(state)
            self.reporter.log("Load optimizer state dict from checkpoint")
        return opt

    def create_scheduler(self,
                         scheduler: str,
                         optimizer: th.optim.Optimizer,
                         state: Optional[Dict] = None,
                         **kwargs):
        """
        Return a learning rate scheduler
        """
        if scheduler not in lr_scheduler_cls:
            raise ValueError(f"Unsupported lr scheduler: {scheduler}")
        lr_scheduler = lr_scheduler_cls[scheduler](optimizer, **kwargs)
        self.reporter.log(f"Create scheduler {scheduler}: {kwargs}")
        if state is not None:
            lr_scheduler.load_state_dict(state)
            self.reporter.log("Load scheduler state dict from checkpoint")
        return lr_scheduler

    def load_checkpoint(
            self,
            cpt_path: str,
            manner: str = "resume") -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Load checkpoint
        """
        if manner not in ["resume", "init"]:
            raise ValueError(f"Unsupported manner: {manner}")
        cpt_stats = th.load(cpt_path, map_location="cpu")
        self.task.nnet.load_state_dict(cpt_stats["model_state_dict"])
        optimizer_dict = None
        if manner == "resume":
            self.reporter.log(f"Resume from checkpoint {cpt_path}: " +
                              f"epoch {self.cur_epoch}")
            optimizer_dict = cpt_stats["optim_state_dict"]
            # set current epoch number
            self.cur_epoch = cpt_stats["epoch"]
        else:
            self.reporter.log(f"Intialized from checkpoint {cpt_path}: " +
                              f"epoch {self.cur_epoch}")
        return cpt_stats, optimizer_dict

    def checkpoint_states(self, epoch: int) -> Dict:
        """
        Return states of the checkpoint to be saved
        """
        raise NotImplementedError

    def save_checkpoint(self, epoch: int, best: bool = True) -> NoReturn:
        """
        Save checkpoint (epoch, model, optimizer, ...)
        """
        if self.rank in [0, None]:
            cpt = self.checkpoint_states(epoch)
            cpt_name = "{}.pt.tar".format("best" if best else "last")
            th.save(cpt, self.checkpoint / cpt_name)
            self.reporter.log(f"Save checkpoint {self.checkpoint / cpt_name}")
            if self.save_interval > 0 and epoch % self.save_interval == 0:
                th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")

    def train_one_step(self, egs: Dict) -> bool:
        """
        Make one training step (return true if no error exists)

        1) Zero optimizer
        2) Forward & Backword
        3) Clip Gradient
        4) Step optimizer
        """
        self.optimizer.zero_grad()

        stats = self.task(egs, ssr=self.ssr)
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
            if self.gaussian_noise_std:
                add_gaussian_noise(self.task, std=self.gaussian_noise_std)
            if norm != -1:
                stats["norm"] = norm
            stats["rate"] = self.optimizer.param_groups[0]["lr"]
            self.reporter.update(stats)
            # schedule lr if needed
            self.lr_scheduler_step(None, end_at="step")
            return True
        else:
            self.reporter.log(f"Invalid gradient {norm:.3f}, skip...")
            return False

    def lr_scheduler_step(self,
                          update_value: Optional[float],
                          end_at: str = "epoch") -> NoReturn:
        """
        Make one step in lr scheduler
        """
        if end_at == "step" and self.lr_scheduler_period == "step":
            self.lr_scheduler.step()
        if end_at == "epoch" and self.lr_scheduler_period == "epoch":
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(update_value)
            else:
                self.lr_scheduler.step()

    def train_epoch(self, data_loader: Iterable[Dict]) -> NoReturn:
        """
        Run one training epoch
        """
        self.task.train()
        self.reporter.train()
        # for idx, egs in enumerate(data_loader):
        for egs in data_loader:
            # load to gpu
            egs = self._prep_egs(egs)
            # make one training step
            self.train_one_step(egs)

    def valid_epoch(self, data_loader: Iterable[Dict]) -> NoReturn:
        """
        Run one validation epoch
        """
        self.task.eval()
        self.reporter.eval()

        with th.no_grad():
            # for idx, egs in enumerate(data_loader):
            for egs in data_loader:
                # load to gpu
                egs = self._prep_egs(egs)
                stats = self.task(egs)
                # update statistics
                self.reporter.update(stats)

    def _prep_egs(self, egs: Dict) -> Dict:
        """
        Prepare training egs
        """
        egs = load_obj(egs, self.default_device)
        # use ssr = 0 when in eval mode
        if self.ss_scheduler:
            egs["ssr"] = self.ssr if self.task.training else 0
        return egs

    def _prep_run(self, dev_loader: Iterable[Dict]) -> int:
        """
        Prepare for training
        """
        # valid
        self.valid_epoch(dev_loader)
        e = self.cur_epoch
        best_loss, best_accu, _ = self.reporter.report(e, 0)
        if self.ss_scheduler:
            self.ssr = self.ss_scheduler.step(e, best_accu)
        # make sure not inf
        best_value = best_loss if self.stop_on == "loss" else best_accu
        # for ReduceLROnPlateau
        if hasattr(self.lr_scheduler, "best"):
            self.lr_scheduler.best = best_value
        self.stop_criterion.reset(best_value)
        # log here
        sstr = f"Epoch {e:d}, loss = {best_loss:.4f}"
        if best_accu is not None:
            sstr += f", accu = {best_accu:.2f}"
        self.reporter.log(sstr)
        return e

    def run(self,
            trn_loader: Iterable[Dict],
            dev_loader: Iterable[Dict],
            num_epochs: int = 50) -> NoReturn:
        """
        Treat whole training set as one training epoch
        """
        self.reporter.log(
            f"Number of batches: {len(trn_loader)}/{len(dev_loader)}")
        e = self._prep_run(dev_loader)
        while e < num_epochs:
            trn_loader.set_epoch(e)
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            # >> train
            self.train_epoch(trn_loader)
            _, _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> valid
            self.valid_epoch(dev_loader)
            cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
            # schedule sampling for eval
            if self.ss_scheduler:
                sstr += f" | ssr = {self.ssr:.3f}"

            update_value = cv_loss if self.stop_on == "loss" else cv_accu
            better = self.stop_criterion.step(update_value)
            if better:
                self.save_checkpoint(e, best=True)
            else:
                sstr += f" | no impr: {self.stop_criterion.no_impr:d}, "
                sstr += f"best = {self.stop_criterion.best:.4f}"

            self.reporter.log(sstr)
            # << valid
            # lr schedule here
            self.lr_scheduler_step(update_value, end_at="epoch")
            if self.ss_scheduler:
                self.ssr = self.ss_scheduler.step(e, cv_accu)
            # save last checkpoint
            self.save_checkpoint(e, best=False)
            # early stop
            if self.stop_criterion.stop():
                self.reporter.log("Stop training cause no impr for " +
                                  f"{self.no_impr} epochs")
                break
        self.reporter.log(f"Training for {e:d}/{num_epochs:d} epochs done!")

    def run_batch_per_epoch(self,
                            trn_loader: Iterable[Dict],
                            dev_loader: Iterable[Dict],
                            num_epochs: int = 100,
                            eval_interval: int = 4000) -> NoReturn:
        """
        For large training set, validate model after several batches
        """
        self.reporter.log(
            f"Number of batches: {len(trn_loader)}/{len(dev_loader)}")
        e = self._prep_run(dev_loader)
        stop = False
        trained_batches = 0
        # set train mode
        self.task.train()
        self.reporter.train()
        while True:
            # trained on several batches
            # for idx, egs in enumerate(trn_loader):
            for egs in trn_loader:
                # update per-batch
                egs = self._prep_egs(egs)
                if self.train_one_step(egs):
                    trained_batches = (trained_batches + 1) % eval_interval

                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    e += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, _, sstr = self.reporter.report(e, cur_lr)
                    self.reporter.log(sstr)

                    self.valid_epoch(dev_loader)
                    cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
                    # schedule sampling for eval
                    if self.ss_scheduler:
                        sstr += f" | ssr = {self.ssr:.3f}"

                    update_value = cv_loss if self.stop_on == "loss" else cv_accu
                    better = self.stop_criterion.step(update_value)
                    if better:
                        self.save_checkpoint(e, best=True)
                    else:
                        sstr += f" | no impr {self.stop_criterion.no_impr:d}, "
                        sstr += f"best = {self.stop_criterion.best:.4f}"
                    self.reporter.log(sstr)
                    # lr schedule here
                    self.lr_scheduler_step(update_value, end_at="epoch")
                    if self.ss_scheduler:
                        self.ssr = self.ss_scheduler.step(e, cv_accu)
                    # save last checkpoint
                    self.save_checkpoint(e, best=False)
                    # reset reporter
                    self.reporter.reset()
                    # early stop or not
                    if self.stop_criterion.stop():
                        self.reporter.log("Stop training cause no impr for " +
                                          f"{self.no_impr} epochs")
                        stop = True
                        break
                    if e == num_epochs:
                        stop = True
                        break
                    # enable train mode
                    self.reporter.log("Set train mode...")
                    self.task.train()
                    self.reporter.train()
            self.reporter.log("Finished one epoch on training set")
            if stop:
                break
        self.reporter.log(f"Training for {e:d}/{num_epochs:d} epochs done!")
