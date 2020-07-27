# wujian@2019

import math
from os import environ
from pathlib import Path
from collections import defaultdict

import torch as th
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from aps.trainer.ss import support_ss_scheduler
from aps.trainer.lr import support_lr_scheduler

from aps.utils import load_obj, get_device_ids, get_logger, SimpleTimer
from aps.task import Task


def add_gaussian_noise(nnet, std=0.075):
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
    def __init__(self, checkpoint, period=100, tensorboard=True, rank=None):
        self.period = period
        if rank is None:
            logger_loc = (checkpoint / "trainer.log").as_posix()
            self.header = "Trainer"
        else:
            logger_loc = (checkpoint / f"trainer.rank{rank}.log").as_posix()
            self.header = f"Rank {rank}"

        self.logger = get_logger(logger_loc, file=True)
        if tensorboard:
            self.board_writer = SummaryWriter(checkpoint)
        else:
            self.board_writer = None
        self.reset()

    def log(self, sstr):
        self.logger.info(f"{self.header}: {sstr}")

    def eval(self):
        self.log(">> Set eval mode ...")
        self.mode = "valid"
        self.reset()

    def train(self):
        self.log(">> Set train mode ...")
        self.mode = "train"
        self.reset()

    def reset(self):
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def update(self, dict_obj):
        if dict_obj is None:
            return
        for key in dict_obj:
            self.add(key, dict_obj[key])

    def add(self, key, value):
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            if key == "rate":
                cur = self.stats[key][-1]
                self.log(f"Processed {N:.2e} batches ({key} = {cur:.3e}) ...")
            else:
                avg = sum(self.stats[key][-self.period:]) / self.period
                self.log(f"Processed {N:.2e} batches ({key} = {avg:+.2f}) ...")

    def report(self, epoch, lr):
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
            hstr = f"Loss/Accu(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: "
            cstr = f"{self.mode} = {loss:.4f}/{accu:.2f}({cost:.2f}m/{N:d})"
        else:
            hstr = f"Loss(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: "
            cstr = f"{self.mode} = {loss:.4f}({cost:.2f}m/{N:d})"
        return loss, accu, hstr + cstr


class StopCriterion(object):
    """
    Early stop of the training
    """
    def __init__(self,
                 no_impr,
                 mode="min",
                 init_criterion=math.inf,
                 no_impr_thres=2e-3):
        self.max_no_impr = no_impr
        self.no_impr = 0
        self.no_impr_thres = no_impr_thres
        self.mode = mode
        self.best_criterion = init_criterion

    def reset(self, update_value):
        self.best_criterion = update_value

    def stop(self):
        return self.no_impr == self.max_no_impr

    @property
    def best(self):
        return self.best_criterion

    def step(self, update_value):
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
        if not isinstance(task, Task):
            raise TypeError(
                f"Trainer accepts Task object, but got {type(task)}")
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

        lr_scheduler_dict = None
        if resume or init:
            cpt_path = resume if resume else init
            if not Path(cpt_path).exists():
                raise FileNotFoundError(
                    f"Could not find checkpoint: {cpt_path}")
            cpt = th.load(cpt_path, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            task.nnet.load_state_dict(cpt["model_state_dict"])
            lr_scheduler_dict = cpt["lr_scheduler_dict"]
            if resume:
                self.reporter.log(f"Resume from checkpoint {cpt_path}: " +
                                  f"epoch {self.cur_epoch}")
                self.optimizer = self.create_optimizer(
                    optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
            else:
                self.reporter.log(f"Intialized from checkpoint {cpt_path}: " +
                                  f"epoch {self.cur_epoch}")
                self.optimizer = self.create_optimizer(optimizer,
                                                       optimizer_kwargs)
        else:
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        if ss_scheduler_kwargs:
            self.ss_scheduler = support_ss_scheduler(ss_scheduler,
                                                     **ss_scheduler_kwargs)
            self.reporter.log(f"Using schedule sampling: {ss_scheduler}")
        else:
            self.ss_scheduler = None

        if lr_scheduler == "reduce_lr":
            self.lr_scheduler = support_lr_scheduler(lr_scheduler,
                                                     self.optimizer,
                                                     mode=mode,
                                                     threshold_mode="abs",
                                                     threshold=no_impr_thres,
                                                     **lr_scheduler_kwargs)
        else:
            self.lr_scheduler = support_lr_scheduler(lr_scheduler,
                                                     self.optimizer,
                                                     mode=mode,
                                                     **lr_scheduler_kwargs)

        if lr_scheduler_dict:
            self.lr_scheduler.load_state_dict(lr_scheduler_dict)

        # logging
        if rank is None:
            self.reporter.log(f"Loading model to GPU:{device_ids[0]}, " +
                              f"#param: {self.num_params:.2f}M")
        else:
            self.reporter.log(
                f"Loading model {rank} to GPU-{rank}/{self.cuda_devices}, " +
                f"#param: {self.num_params:.2f}M")

        self.reporter.log(f"Schedule sampling strategy: {ss_scheduler}")
        self.reporter.log(f"Stop criterion: {self.stop_on}")
        if clip_gradient:
            self.reporter.log(
                f"Gradient clipping if over {clip_gradient} L2 norm")
        if gaussian_noise_std:
            self.reporter.log("Add gaussian noise to weights, with " +
                              f"std = {gaussian_noise_std}")

    def create_optimizer(self, optimizer, kwargs, state=None):
        """
        Return a pytorch-optimizer
        """
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax,  # lr, weight_decay
            "adamw": th.optim.AdamW,  # lr, weight_decay
            # NOTE: move to lr scheduler
            # "noam": NoamOpt
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

    def save_checkpoint(self, epoch, best=True):
        """
        Save checkpoint (epoch, model, optimizer)
        """
        raise NotImplementedError

    def train(self, data_loader):
        self.task.train()
        self.reporter.train()
        # for idx, egs in enumerate(data_loader):
        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.default_device)

            self.optimizer.zero_grad()

            loss, stats = self.task(egs, ssr=self.ssr)
            loss.backward()

            # add to reporter
            self.reporter.add("loss", loss.item())
            self.reporter.update(stats)

            # clip gradient after backward
            norm = -1
            if self.clip_gradient:
                norm = clip_grad_norm_(self.task.parameters(),
                                       self.clip_gradient)

            loss = loss.item()
            if math.isfinite(norm) and math.isfinite(loss):
                self.optimizer.step()

                if self.gaussian_noise_std:
                    add_gaussian_noise(self.task, std=self.gaussian_noise_std)

                self.reporter.add("norm", norm)
                self.reporter.add("rate", self.optimizer.param_groups[0]["lr"])
            else:
                self.reporter.log(f"Invalid gradient {norm:.3f} or " +
                                  f"loss {loss:.3f}, skip...")

    def eval(self, data_loader):
        self.task.eval()
        self.reporter.eval()

        with th.no_grad():
            # for idx, egs in enumerate(data_loader):
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                # ssr = 0, use ground truth
                loss, stats = self.task(egs, ssr=0)
                self.reporter.add("loss", loss.item())
                self.reporter.update(stats)

    def _prep_train(self, dev_loader):
        """
        Prepare for training
        """
        # eval
        self.eval(dev_loader)
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

    def run(self, trn_loader, dev_loader, num_epochs=50):
        """
        Run on whole training set and evaluate
        """
        self.reporter.log(
            f"Number of batches (train/valid) = {len(trn_loader)}/{len(dev_loader)}"
        )
        e = self._prep_train(dev_loader)
        while e < num_epochs:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            # >> train
            self.train(trn_loader)
            _, _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(dev_loader)
            cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
            # schedule sampling for eval
            if self.ss_scheduler:
                sstr += f" | ssr = {self.ssr:.3f}"

            update_value = cv_loss if self.stop_on == "loss" else cv_accu
            better = self.stop_criterion.step(update_value)
            if better:
                self.save_checkpoint(e, best=True)
            else:
                sstr += f" | no impr, best = {self.stop_criterion.best:.4f}"

            self.reporter.log(sstr)
            # << eval
            # schedule here
            if self.lr_scheduler:
                self.lr_scheduler.step(update_value)
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
                            trn_loader,
                            dev_loader,
                            num_epochs=100,
                            eval_interval=4000):
        """
        Run on several batches and evaluate
        """
        self.reporter.log("Number of batches (train/valid) = " +
                          f"{len(trn_loader)}/{len(dev_loader)}")
        e = self._prep_train(dev_loader)
        stop = False
        trained_batches = 0
        # set train mode
        self.task.train()
        self.reporter.train()
        while True:
            # trained on several batches
            # for idx, egs in enumerate(trn_loader):
            for egs in trn_loader:
                trained_batches = (trained_batches + 1) % eval_interval
                # update per-batch
                egs = load_obj(egs, self.default_device)
                self.optimizer.zero_grad()

                loss, stats = self.task(egs, ssr=self.ssr)
                loss.backward()

                # add to reporter
                self.reporter.add("loss", loss.item())
                self.reporter.update(stats)

                norm = -1
                if self.clip_gradient:
                    norm = clip_grad_norm_(self.task.parameters(),
                                           self.clip_gradient)
                loss = loss.item()
                if math.isfinite(norm) and math.isfinite(loss):
                    self.optimizer.step()

                    if self.gaussian_noise_std:
                        add_gaussian_noise(self.task,
                                           std=self.gaussian_noise_std)

                    self.reporter.add("norm", norm)
                    self.reporter.add("rate",
                                      self.optimizer.param_groups[0]["lr"])
                else:
                    self.reporter.log(f"Invalid gradient {norm:.3f} or " +
                                      f"loss {loss:.3f}, skip...")

                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    e += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, _, sstr = self.reporter.report(e, cur_lr)
                    self.reporter.log(sstr)

                    self.eval(dev_loader)
                    cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
                    # schedule sampling for eval
                    if self.ss_scheduler:
                        sstr += f" | ssr = {self.ssr:.3f}"

                    update_value = cv_loss if self.stop_on == "loss" else cv_accu
                    better = self.stop_criterion.step(update_value)
                    if better:
                        self.save_checkpoint(e, best=True)
                    else:
                        sstr += f" | no impr, best = {self.stop_criterion.best:.4f}"

                    self.reporter.log(sstr)
                    # schedule here
                    if self.lr_scheduler:
                        self.lr_scheduler.step(update_value)
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