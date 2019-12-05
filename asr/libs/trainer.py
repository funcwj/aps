# wujian@2019

import sys
import math
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

from .utils import get_logger, load_obj, get_device_ids, SimpleTimer
from .scheduler import support_ss_scheduler

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


def add_gaussian_noise(nnet, std=0.075):
    """
    Add gaussian noise to updated weights
    """
    for p in nnet.parameters():
        if p.requires_grad:
            noise = th.randn(p.data.shape, device=nnet.device)
            p.data += noise * std


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

        if N == 0:
            raise RuntimeError("No statistics to report")
        loss = sum(self.stats["loss"]) / N
        accu = sum(self.stats["accu"]) * 100 / N
        if self.board_writer:
            self.board_writer.add_scalar(f"loss/{self.mode}", loss, epoch)
            self.board_writer.add_scalar(f"accu/{self.mode}", accu, epoch)
        cost = self.timer.elapsed()
        hstr = f"Loss/Accu(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: "
        cstr = f"{self.mode} = {loss:.4f}/{accu:.2f}({cost:.2f}m/{N:d})"
        return loss, accu, hstr + cstr


class EarlyStopCriterion(object):
    """
    Early stop of the training
    """
    def __init__(self, no_impr, order="min", init_criterion=math.inf):
        self.max_no_impr = no_impr
        self.no_impr = 0
        self.order = order
        self.best_criterion = init_criterion

    def reset(self, update_value):
        self.best_criterion = update_value

    def stop(self):
        return self.no_impr == self.max_no_impr

    def step(self, update_value):
        c1 = self.best_criterion < update_value and self.order == "min"
        c2 = self.best_criterion > update_value and self.order == "max"
        if c1 or c2:
            self.no_impr += 1
            return False
        else:
            self.best_criterion = update_value
            self.no_impr = 0
            return True


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
                 ss_scheduler="const",
                 ss_prob=0,
                 ss_scheduler_kwargs=None,
                 gradient_clip=None,
                 gaussian_noise=None,
                 prog_interval=100,
                 save_interval=-1,
                 resume="",
                 init="",
                 tensorboard=False,
                 no_impr=6):
        self.device_ids = get_device_ids(device_ids)
        self.default_device = th.device(f"cuda:{self.device_ids[0]}")

        self.checkpoint = Path(checkpoint)
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self.reporter = ProgressReporter(self.checkpoint,
                                         period=prog_interval,
                                         tensorboard=tensorboard)

        self.gradient_clip = gradient_clip
        self.gaussian_noise = gaussian_noise
        self.cur_epoch = 0  # zero based
        self.save_interval = save_interval
        self.lsm_factor = lsm_factor
        self.ssr = 0

        self.early_stop = EarlyStopCriterion(no_impr)
        self.ss_scheduler = support_ss_scheduler(ss_scheduler, ss_prob,
                                                 **ss_scheduler_kwargs)

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
        self.reporter.log(f"Schedule sampling strategy: {ss_scheduler}")
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

            loss, accu = self.compute_loss(egs, idx=idx, ssr=self.ssr)
            loss.backward()

            # clip gradient after backward
            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)

            loss_value = loss.item()
            if math.isinf(norm) or math.isnan(loss_value):
                self.reporter.log(f"Invalid gradient {norm} or " +
                                  f"loss {loss_value}, skip...")
            else:
                self.optimizer.step()

                if self.gaussian_noise:
                    add_gaussian_noise(self.nnet, std=self.gaussian_noise)

                self.reporter.add("norm", norm)
                self.reporter.add("loss", loss_value)
                self.reporter.add("accu", accu)

    def eval(self, data_loader):
        self.nnet.eval()
        self.reporter.eval()

        with th.no_grad():
            for idx, egs in enumerate(data_loader):
                egs = load_obj(egs, self.default_device)
                # ssr = 0, use ground truth
                loss, accu = self.compute_loss(egs, idx=idx, ssr=0)
                self.reporter.add("loss", loss.item())
                self.reporter.add("accu", accu)

    def run(self, train_loader, valid_loader, num_epoches=50):
        """
        Run on whole training set and evaluate
        """
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        # make dilated conv faster
        th.backends.cudnn.benchmark = True

        self.eval(valid_loader)
        e = self.cur_epoch
        best_loss, best_accu, _ = self.reporter.report(e, 0)
        self.ssr = self.ss_scheduler.step(e, best_accu)
        # make sure not inf
        self.scheduler.best = best_loss
        self.early_stop.reset(best_loss)

        self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
        while e < num_epoches:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            # >> train
            self.train(train_loader)
            _, _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(valid_loader)
            cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
            # schedule sampling for eval
            sstr += f" | ssr = {self.ssr:.3f}"

            better = self.early_stop.step(cv_loss)
            if better:
                self.save_checkpoint(e, best=True)
            else:
                sstr += f" | no impr, best = {self.scheduler.best:.4f}"

            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(cv_loss)
            self.ssr = self.ss_scheduler.step(e, cv_accu)
            # flush scheduler info
            sys.stdout.flush()
            # save last checkpoint
            self.save_checkpoint(e, best=False)
            # early stop
            if self.early_stop.stop():
                self.reporter.log("Stop training cause no impr for " +
                                  f"{self.early_stop.no_impr:d} epochs")
                break
        self.reporter.log(f"Training for {e:d}/{num_epoches:d} epoches done!")

    def run_batch_per_epoch(self,
                            train_loader,
                            valid_loader,
                            num_epoches=100,
                            eval_interval=4000):
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        # make dilated conv faster
        th.backends.cudnn.benchmark = True

        e = self.cur_epoch
        self.eval(valid_loader)

        best_loss, best_accu, _ = self.reporter.report(e, 0)
        self.ssr = self.ss_scheduler.step(e, best_accu)
        self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
        # make sure not inf
        self.scheduler.best = best_loss
        self.early_stop.reset(best_loss)

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

                loss, accu = self.compute_loss(egs, idx=idx, ssr=self.ssr)
                loss.backward()

                if self.gradient_clip:
                    norm = clip_grad_norm_(self.nnet.parameters(),
                                           self.gradient_clip)
                loss_value = loss.item()
                if math.isinf(norm) or math.isnan(loss_value):
                    self.reporter.log(f"Invalid gradient {norm} or " +
                                      f"loss {loss_value}, skip...")
                else:
                    self.optimizer.step()

                    self.reporter.add("norm", norm)
                    self.reporter.add("loss", loss_value)
                    self.reporter.add("accu", accu)

                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    e += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, _, sstr = self.reporter.report(e, cur_lr)
                    self.reporter.log(sstr)

                    cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
                    # schedule sampling for eval
                    sstr += f" | ssr = {self.ssr:.3f}"

                    better = self.early_stop.step(cv_loss)
                    if better:
                        self.save_checkpoint(e, best=True)
                    else:
                        sstr += f" | no impr, best = {self.scheduler.best:.4f}"

                    self.reporter.log(sstr)
                    # schedule here
                    self.scheduler.step(cv_loss)
                    self.ssr = self.ss_scheduler.step(e, cv_accu)
                    # flush scheduler info
                    sys.stdout.flush()
                    # save last checkpoint
                    self.save_checkpoint(e, best=False)
                    # reset reporter
                    self.reporter.reset()
                    # early stop or not
                    if self.early_stop.stop():
                        self.reporter.log(
                            "Stop training cause no impr for " +
                            f"{self.early_stop.no_impr:d} epochs")
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
    def __init__(self, nnet, ctc_coeff=0, ctc_blank=0, **kwargs):
        super(S2STrainer, self).__init__(nnet, **kwargs)
        self.ctc_coeff = ctc_coeff
        self.ctc_blank = ctc_blank
        if ctc_coeff:
            self.reporter.log("Using CTC regularization (coeff = " +
                              f"{ctc_coeff:.2f}, blank = {ctc_blank})")

    def compute_loss(self, egs, idx=0, ssr=0):
        """
        Compute training loss, egs contains
            src_pad: N x Ti x F
            src_len: N
            tgt_pad: N x To
            tgt_len: N
        """
        # N x To, -1 => EOS
        ignored_mask = egs["tgt_pad"] == IGNORE_ID
        tgt_pad = egs["tgt_pad"].masked_fill(ignored_mask, self.nnet.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        outs, _, ctc_branch, enc_len = data_parallel(
            self.nnet, (egs["src_pad"], egs["src_len"], tgt_pad, ssr),
            device_ids=self.device_ids)
        # N x (To+1), pad -1
        tgts = F.pad(egs["tgt_pad"], (0, 1), value=IGNORE_ID)
        # add eos
        tgts = tgts.scatter(1, egs["tgt_len"][:, None], self.nnet.eos)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_loss(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_loss(outs, tgts)

        if self.ctc_coeff > 0:
            # add log-softmax, N x T x V => T x N x V
            log_prob = F.log_softmax(ctc_branch, dim=-1).transpose(0, 1)
            ctc_loss = F.ctc_loss(log_prob,
                                  tgt_pad,
                                  enc_len,
                                  egs["tgt_len"],
                                  blank=self.ctc_blank,
                                  reduction="mean",
                                  zero_infinity=True)
            loss = self.ctc_coeff * ctc_loss + (1 - self.ctc_coeff) * loss
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

    def compute_loss(self, egs, idx=0, ssr=0):
        """
        Compute training loss, egs contains
            src: N x T
            tgt: N x T
        """
        if idx == 0:
            self.hidden = None
        # pred: N x T x V
        pred, self.hidden = data_parallel(self.nnet, (egs["src"], self.hidden),
                                          device_ids=self.device_ids)
        loss = ce_loss(pred, egs["tgt"])
        accu = compute_accu(pred, egs["tgt"])

        return loss, accu
