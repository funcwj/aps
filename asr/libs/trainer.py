# wujian@2019

import sys
import uuid
import random

from pathlib import Path
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import data_parallel
from torch.utils.tensorboard import SummaryWriter
from torch import autograd

from .utils import get_logger, load_obj, get_device_ids, add_gaussian_noise, SimpleTimer

IGNORE_ID = -1


def ce_loss(outs, tgts):
    """
    Cross entropy loss
    """
    _, _, V = outs.shape
    # N(To+1) x V
    outs = outs.view(-1, V)
    # N(To+1)
    tgts = tgts.view(-1)
    ce_loss = F.cross_entropy(outs, tgts, ignore_index=-1, reduction="mean")
    return ce_loss


def ls_loss(outs, tgts, lsm_factor=0.1):
    """
    Label smooth loss (using KL)
    """
    _, _, V = outs.shape
    # NT x V
    outs = outs.view(-1, V)
    # NT
    tgts = tgts.view(-1)
    mask = (tgts != IGNORE_ID)
    # M x V
    outs = th.masked_select(outs, mask.unsqueeze(-1)).view(-1, V)
    # M
    tgts = th.masked_select(tgts, mask)
    # M x V
    dist = outs.new_full(outs.size(), lsm_factor / (V - 1))
    dist = dist.scatter_(1, tgts.unsqueeze(-1), 1 - lsm_factor)
    # KL distance
    loss = F.kl_div(F.log_softmax(outs, -1), dist, reduction="batchmean")
    return loss


def compute_accu(outs, tgts):
    """
    Compute frame-level accuracy
    """
    # N x (To+1)
    pred = th.argmax(outs.detach(), dim=-1)
    # ignore mask, -1
    mask = (tgts != IGNORE_ID)
    ncorr = th.sum(pred[mask] == tgts[mask]).item()
    total = th.sum(mask).item()
    return float(ncorr) / total


class ProgressReporter(object):
    """
    A simple progress reporter
    """
    def __init__(self, checkpoint, period=100, tensorboard=True):
        self.period = period
        logger_loc = (checkpoint / "trainer.log").as_posix()
        self.logger = get_logger(logger_loc, file=True)
        if tensorboard:
            self.board_writer = SummaryWriter(checkpoint)
        else:
            self.board_writer = None
        self.header = "Trainer"
        self.reset()

    def log(self, sstr):
        self.logger.info(f"{self.header}: {sstr}")

    def eval(self):
        self.log("set eval mode...")
        self.mode = "valid"
        self.reset()

    def train(self):
        self.log("set train mode...")
        self.mode = "train"
        self.reset()

    def reset(self):
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def add(self, key, value):
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            avg = sum(self.stats[key][-self.period:]) / self.period
            self.log(f"processed {N:.2e} batches ({key} = {avg:+.2f})...")

    def report(self, epoch, lr):
        N = len(self.stats["loss"])
        if self.mode == "valid":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"loss on {N:d} batches: {sstr}")

        loss = sum(self.stats["loss"]) / N
        accu = sum(self.stats["accu"]) * 100 / N
        if self.board_writer:
            self.board_writer.add_scalar(f"loss/{self.mode}", loss, epoch)
            self.board_writer.add_scalar(f"accu/{self.mode}", accu, epoch)
        cost = self.timer.elapsed()
        hstr = f"Loss/Accu(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: "
        cstr = f"{self.mode} = {loss:.4f}/{accu:.2f}({cost:.2f}m/{N:d})"
        return loss, hstr + cstr


class Trainer(object):
    """
    A PyTorch base trainer
    """
    def __init__(self,
                 nnet,
                 checkpoint="cpt",
                 optimizer="adam",
                 device_ids=0,
                 optimizer_kwargs=None,
                 lr_scheduler_kwargs=None,
                 lsm_factor=0,
                 schedule_sampling=0,
                 schedule_strategy="const",
                 gradient_clip=None,
                 gaussian_noise=None,
                 logging_period=100,
                 save_interval=-1,
                 resume="",
                 init="",
                 tensorboard=False,
                 no_impr=6):
        if schedule_strategy not in ["const", "saturate"]:
            raise ValueError(f"Unknown schedule strategy: {schedule_strategy}")
        self.device_ids = get_device_ids(device_ids)
        self.default_device = th.device(f"cuda:{self.device_ids[0]}")

        self.checkpoint = Path(checkpoint)
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self.reporter = ProgressReporter(self.checkpoint,
                                         period=logging_period,
                                         tensorboard=tensorboard)

        self.gradient_clip = gradient_clip
        self.gaussian_noise = gaussian_noise
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.save_interval = save_interval
        self.lsm_factor = lsm_factor
        self.ssr_init = schedule_sampling
        self.ssr_vary = 0
        self.sss = schedule_strategy

        if resume or init:
            cpt_path = resume if resume else init
            if not Path(cpt_path).exists():
                raise FileNotFoundError(
                    f"Could not find checkpoint: {cpt_path}")
            cpt = th.load(cpt_path, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.default_device)
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
            self.nnet = nnet.to(self.default_device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           **lr_scheduler_kwargs)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.reporter.log(f"Model summary:\n{nnet}")
        self.reporter.log(f"Loading model to GPUs:{self.device_ids}, " +
                          f"#param: {self.num_params:.2f}M")
        if gradient_clip:
            self.reporter.log(
                f"Gradient clipping by {gradient_clip}, default L2")
        if gaussian_noise:
            self.reporter.log(
                f"Add gaussian noise to weights with std = {gaussian_noise}")

    def save_checkpoint(self, epoch, best=True):
        """
        Save checkpoint (epoch, model, optimizer)
        """
        cpt = {
            "epoch": epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        cpt_name = "{}.pt.tar".format("best" if best else "last")
        th.save(cpt, self.checkpoint / cpt_name)
        self.reporter.log(f"save checkpoint {cpt_name}")
        if self.save_interval > 0 and epoch % self.save_interval == 0:
            th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")

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
            "adamw": th.optim.AdamW  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError(f"Now only support optimizer {optimizer}")
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.reporter.log(f"Create optimizer {optimizer}: {kwargs}")
        if state is not None:
            opt.load_state_dict(state)
            self.reporter.log("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs, **kwargs):
        """
        Compute training loss, return loss and other numbers
        """
        raise NotImplementedError

    def train(self, data_loader):
        self.nnet.train()
        self.reporter.train()

        for idx, egs in enumerate(data_loader):
            # load to gpu
            egs = load_obj(egs, self.default_device)

            self.optimizer.zero_grad()
            ssr = self.ssr_init if self.sss == "const" else self.ssr_vary
            loss, accu = self.compute_loss(egs, idx=idx, ssr=ssr)
            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)

            if float("inf") == norm or th.isnan(loss).item():
                self.reporter.log("Invalid gradient or loss, skip...")
            else:
                loss.backward()
                self.optimizer.step()

                if self.gaussian_noise:
                    add_gaussian_noise(self.nnet, std=self.gaussian_noise)

                self.reporter.add("norm", norm)
                self.reporter.add("loss", loss.item())
                self.reporter.add("accu", accu)

    def eval(self, data_loader):
        self.nnet.eval()
        self.reporter.eval()

        with th.no_grad():
            for idx, egs in enumerate(data_loader):
                egs = load_obj(egs, self.default_device)
                # eval, ssr=0
                loss, accu = self.compute_loss(egs, idx=idx, ssr=0)
                self.reporter.add("loss", loss.item())
                self.reporter.add("accu", accu)

    def run(self, train_loader, valid_loader, num_epoches=50):
        """
        Run on whole training set and evaluate
        """
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        # check if save is OK
        self.save_checkpoint(0, best=False)

        self.eval(valid_loader)
        e = self.cur_epoch
        best_loss, _ = self.reporter.report(e, 0)

        self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
        # make sure not inf
        self.scheduler.best = best_loss
        no_impr = 0

        while e < num_epoches:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            # >> train
            self.train(train_loader)
            _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(valid_loader)
            cv_loss, sstr = self.reporter.report(e, cur_lr)
            # schedule sampling for eval
            rate = self.ssr_init if self.sss == "const" else self.ssr_vary
            sstr += f" | ssr = {rate:.3f}"
            if cv_loss > best_loss:
                no_impr += 1
                if self.ssr_vary != self.ssr_init:
                    self.ssr_vary = self.ssr_init
                sstr += f" | no impr, best = {self.scheduler.best:.4f}"
            else:
                best_loss = cv_loss
                no_impr = 0
                self.save_checkpoint(e, best=True)
            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(cv_loss)
            # flush scheduler info
            sys.stdout.flush()
            # save last checkpoint
            self.save_checkpoint(e, best=False)
            if no_impr == self.no_impr:
                self.reporter.log(
                    f"Stop training cause no impr for {no_impr:d} epochs")
                break
        self.reporter.log(f"Training for {e:d}/{num_epoches:d} epoches done!")

    def run_batch_per_epoch(self,
                            train_loader,
                            valid_loader,
                            num_epoches=100,
                            eval_interval=4000):
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)

        e = self.cur_epoch
        self.eval(valid_loader)
        best_loss, _ = self.reporter.report(e, 0)
        self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
        # make sure not inf
        self.scheduler.best = best_loss
        no_impr = 0

        stop = False
        trained_batches = 0
        # set train mode
        self.nnet.train()
        self.reporter.train()
        while True:
            # trained on several batches
            for idx, egs in enumerate(train_loader):
                trained_batches = (trained_batches + 1) % eval_interval
                # update per-batch
                egs = load_obj(egs, self.default_device)
                self.optimizer.zero_grad()

                ssr = self.ssr_init if self.sss == "const" else self.ssr_vary
                loss, accu = self.compute_loss(egs, idx=idx, ssr=ssr)
                if self.gradient_clip:
                    norm = clip_grad_norm_(self.nnet.parameters(),
                                           self.gradient_clip)

                if float("inf") == norm or th.isnan(loss).item():
                    self.reporter.log("Invalid gradient or loss, skip...")
                else:
                    loss.backward()
                    self.optimizer.step()

                    self.reporter.add("norm", norm)
                    self.reporter.add("loss", loss.item())
                    self.reporter.add("accu", accu)

                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    e += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, sstr = self.reporter.report(e, cur_lr)
                    self.reporter.log(sstr)

                    cv_loss, sstr = self.reporter.report(e, cur_lr)
                    # schedule sampling for eval
                    rate = self.ssr_init if self.sss == "const" else self.ssr_vary
                    sstr += f" | ssr = {rate:.3f}"

                    if cv_loss > best_loss:
                        no_impr += 1
                        sstr += f" | no impr, best = {self.scheduler.best:.4f}"
                        if self.ssr_vary != self.ssr_init:
                            self.ssr_vary = self.ssr_init
                    else:
                        best_loss = cv_loss
                        no_impr = 0
                        self.save_checkpoint(e, best=True)
                    self.reporter.log(sstr)
                    # schedule here
                    self.scheduler.step(cv_loss)
                    # flush scheduler info
                    sys.stdout.flush()
                    # save last checkpoint
                    self.save_checkpoint(e, best=False)
                    # reset reporter
                    self.reporter.reset()
                    # early stop or not
                    if no_impr == self.no_impr:
                        self.reporter.log("Stop training cause no impr " +
                                          f"for {no_impr:d} epochs")
                        stop = True
                        break
                    if e == num_epoches:
                        stop = True
                        break
                    # enable train mode
                    self.reporter.log("Set train mode...")
                    self.nnet.train()
            self.reporter.log("Finished one epoch on training set")
            if stop:
                break
        self.reporter.log(f"Training for {e:d}/{num_epoches:d} epoches done!")


class S2STrainer(Trainer):
    """
    E2E ASR Trainer (CE)
    """
    def __init__(self, *args, **kwargs):
        super(S2STrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, egs, idx=0, ssr=0):
        """
        Compute training loss, egs contains
            x_pad: N x Ti x F
            x_len: N
            y_pad: N x To
            y_len: N
        """
        # N x To, -1 => EOS
        y_pad = egs["y_pad"].masked_fill(egs["y_pad"] == IGNORE_ID,
                                         self.nnet.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        outs, _ = data_parallel(self.nnet,
                                (egs["x_pad"], egs["x_len"], y_pad, ssr),
                                device_ids=self.device_ids)
        # N x (To+1), pad -1
        tgts = F.pad(egs["y_pad"], (0, 1), value=IGNORE_ID)
        # add eos
        tgts = tgts.scatter(1, egs["y_len"][:, None], self.nnet.eos)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_loss(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_loss(outs, tgts)
        # compute accu
        accu = compute_accu(outs, tgts)
        return loss, accu


class LmTrainer(Trainer):
    """
    E2E ASR Trainer (CE)
    """
    def __init__(self, *args, **kwargs):
        super(LmTrainer, self).__init__(*args, **kwargs)
        self.hidden = None
        if self.sss != "const" and self.ssr_init != 0:
            raise RuntimeError(
                "For LM training, we don't use schedule sampling")

    def compute_loss(self, egs, idx=0, ssr=0):
        """
        Compute training loss, egs contains
            x: N x T
            y: N x T
        """
        if idx == 0:
            self.hidden = None
        # pred: N x T x V
        pred, self.hidden = data_parallel(self.nnet, (egs["x"], self.hidden),
                                          device_ids=self.device_ids)
        loss = ce_loss(pred, egs["y"])
        accu = compute_accu(pred, egs["y"])

        return loss, accu
