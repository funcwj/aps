# wujian@2018

import os
import sys
import time
import random
import uuid

from itertools import permutations
from collections import defaultdict

import numpy as np
import torch as th
import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch import autograd

from .logger import get_logger


def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def get_device_ids(device_ids):
    """
    Got device ids
    """
    if not th.cuda.is_available():
        raise RuntimeError("CUDA device unavailable...exist")
    if device_ids is None:
        # detect number of device available
        dev_cnt = th.cuda.device_count()
        device_ids = tuple(range(0, dev_cnt))
    if isinstance(device_ids, int):
        device_ids = (device_ids, )
    return device_ids


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.reset()

    def reset(self):
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def add(self, key, value):
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            avg = sum(self.stats[key][-self.period:]) / self.period
            self.logger.info("Processed {:.2e} batches "
                             "({} = {:+.2f})...".format(N, key, avg))

    def report(self, details=False):
        N = len(self.stats["loss"])
        if details:
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["accu"]))
            self.logger.info("Accu on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.stats["loss"]) / N,
            "accu": sum(self.stats["accu"]) * 100 / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }


class S2STrainer(object):
    """
    A PyTorch seq2seq trainer
    """

    def __init__(self,
                 nnet,
                 checkpoint="cpt",
                 optimizer="adam",
                 device_ids=0,
                 optimizer_kwargs=None,
                 lr_scheduler_kwargs=None,
                 label_smooth=0,
                 schedule_sampling=0,
                 gradient_clip=None,
                 logger=None,
                 logging_period=100,
                 resume=None,
                 no_impr=6):
        self.device_ids = get_device_ids(device_ids)
        self.default_device = th.device("cuda:{}".format(self.device_ids[0]))
        if checkpoint:
            os.makedirs(checkpoint, exist_ok=True)
        self.checkpoint = checkpoint
        self.logger = logger if logger else get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.gradient_clip = gradient_clip
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.label_smooth = label_smooth
        self.ssr = schedule_sampling

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = nnet.to(self.default_device)
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            self.nnet = nnet.to(self.default_device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           **lr_scheduler_kwargs)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            device_ids, self.num_params))
        if gradient_clip:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(gradient_clip))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def _dump_ali(self, dump_dir, key, alis, feats=None):
        """
        Sample and dump out alignments
        """
        os.makedirs(dump_dir, exist_ok=True)
        N, _, _ = alis.shape
        n = random.randint(0, N - 1)
        a = alis[n].detach().cpu().numpy()
        # s = np.arange(0, T, 10)
        np.save(os.path.join(dump_dir, key) + "-align", a)
        if feats is not None:
            np.save(
                os.path.join(dump_dir, key) + "-feats",
                feats[n].detach().cpu().numpy())

    def compute_loss(self, egs, ssr=0, dump_ali=False):
        """
        Compute training loss, egs contains
            x_pad: N x Ti x F
            x_len: N
            y_pad: N x To
            y_len: N
        """
        # N x To, -1 => EOS
        ignore_mask = (egs["y_pad"] == -1)
        y_pad = egs["y_pad"].masked_fill(ignore_mask, self.nnet.eos)
        # outs, alis
        # N x (To+1) x V
        # N x (To+1) x Ti
        outs, alis = th.nn.parallel.data_parallel(
            self.nnet, (egs["x_pad"], egs["x_len"], y_pad, ssr),
            device_ids=self.device_ids)
        # outs, alis = self.nnet(egs["x_pad"], egs["x_len"], y_pad, ssr)

        if dump_ali:
            self._dump_ali(os.path.join(self.checkpoint, str(self.cur_epoch)),
                           str(uuid.uuid4()), alis)
        # N x (To+1), pad -1
        tgts = F.pad(egs["y_pad"], (0, 1), value=-1)
        # add eos
        for i in range(tgts.shape[0]):
            tgts[i, egs["y_len"][i]] = self.nnet.eos
        # compute loss
        # if self.label_smooth > 0:
        #     loss = self._ls_loss(outs, tgts)
        # else:
        loss = self._ce_loss(outs, tgts)
        # compute accu
        accu = self._compute_accu(outs, tgts)
        return loss, accu

    def _compute_accu(self, outs, tgts):
        """
        Compute frame-level accuracy
        """
        # N x (To+1)
        pred = th.argmax(outs.detach(), dim=-1)
        # ignore mask, -1
        mask = (tgts != -1)
        ncorr = th.sum(pred[mask] == tgts[mask]).item()
        total = th.sum(mask).item()
        return float(ncorr) / total

    def _ce_loss(self, outs, tgts):
        """
        Cross entropy loss
        """
        _, _, V = outs.shape
        # N(To+1) x V
        outs = outs.view(-1, V)
        # N(To+1)
        tgts = tgts.view(-1)
        loss = F.cross_entropy(outs, tgts, ignore_index=-1, reduction="mean")
        return loss

    def _ls_loss(self, outs, tgts):
        """
        Label smooth loss
        """
        _, _, V = outs.shape
        # N x T x V
        dist = outs.new_full(outs.size(), self.label_smooth / (V - 1))
        dist = dist.scatter_(2, tgts.unsqueeze(2), 1 - self.label_smooth)
        # KL distance
        loss = F.kl_div(F.log_softmax(outs, 2), dist, reduction="mean")
        return loss

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.default_device)

            self.optimizer.zero_grad()
            loss, accu = self.compute_loss(egs, ssr=self.ssr)
            loss.backward()
            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)
                reporter.add("norm", norm)
            self.optimizer.step()

            reporter.add("loss", loss.item())
            reporter.add("accu", accu)
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                # TODO: make ssr=0 when eval?
                loss, accu = self.compute_loss(egs,
                                               ssr=self.ssr,
                                               dump_ali=False)
                reporter.add("loss", loss.item())
                reporter.add("accu", accu)
        return reporter.report(details=True)

    def run(self, train_loader, valid_loader, num_epoches=50):
        """
        Run on whole training set and evaluate
        """
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        stats = dict()
        # check if save is OK
        self.save_checkpoint(best=False)
        cv = self.eval(valid_loader)
        best_loss = cv["loss"]
        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self.cur_epoch, best_loss))
        no_impr = 0
        # make sure not inf
        self.scheduler.best = best_loss
        while self.cur_epoch < num_epoches:
            self.cur_epoch += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            stats["title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                cur_lr, self.cur_epoch)
            tr = self.train(train_loader)
            stats["tr"] = "train = {:.4f}/{:.2f}%({:.2f}m/{:d})".format(
                tr["loss"], tr["accu"], tr["cost"], tr["batches"])
            cv = self.eval(valid_loader)
            stats["cv"] = "dev = {:.4f}/{:.2f}%({:.2f}m/{:d})".format(
                cv["loss"], cv["accu"], cv["cost"], cv["batches"])
            stats["scheduler"] = ""
            if cv["loss"] > best_loss:
                no_impr += 1
                stats["scheduler"] = "| no impr, best = {:.4f}".format(
                    self.scheduler.best)
            else:
                best_loss = cv["loss"]
                no_impr = 0
                self.save_checkpoint(best=True)
            self.logger.info("{title} {tr} | {cv} {scheduler}".format(**stats))
            # schedule here
            self.scheduler.step(cv["loss"])
            # flush scheduler info
            sys.stdout.flush()
            # save last checkpoint
            self.save_checkpoint(best=False)
            if no_impr == self.no_impr:
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(
                        no_impr))
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epoches))

    def run_batch_per_epoch(self,
                            train_loader,
                            valid_loader,
                            num_epoches=100,
                            eval_interval=4000):
        self.logger.info("Number of batches in train/valid = {:d}/{:d}".format(
            len(train_loader), len(valid_loader)))
        stats = dict()
        # make dilated conv faster
        # th.backends.cudnn.benchmark = True
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        # check if save is OK
        self.save_checkpoint(best=False)
        cv = self.eval(valid_loader)
        best_loss = cv["loss"]
        self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
            self.cur_epoch, best_loss))
        no_impr = 0
        stop = False
        trained_batches = 0
        train_reporter = ProgressReporter(self.logger,
                                          period=self.logging_period)
        # make sure not inf
        self.scheduler.best = best_loss
        # set train mode
        self.nnet.train()
        while True:
            # trained on several batches
            for egs in train_loader:
                trained_batches = (trained_batches + 1) % eval_interval
                # update per-batch
                egs = load_obj(egs, self.default_device)
                self.optimizer.zero_grad()
                loss, accu = self.compute_loss(egs, ssr=self.ssr)
                loss.backward()
                if self.gradient_clip:
                    norm = clip_grad_norm_(self.nnet.parameters(),
                                           self.gradient_clip)
                    train_reporter.add("norm", norm)
                self.optimizer.step()
                # record loss & accu
                train_reporter.add("accu", accu)
                train_reporter.add("loss", loss.item())
                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    self.cur_epoch += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    stats[
                        "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                            cur_lr, self.cur_epoch)
                    tr = train_reporter.report()
                    stats[
                        "tr"] = "train = {:.4f}/{:.2f}%({:.2f}m/{:d})".format(
                            tr["loss"], tr["accu"], tr["cost"], tr["batches"])
                    cv = self.eval(valid_loader)
                    stats["cv"] = "dev = {:.4f}/{:.2f}%({:.2f}m/{:d})".format(
                        cv["loss"], cv["accu"], cv["cost"], cv["batches"])
                    stats["scheduler"] = ""
                    if cv["loss"] > best_loss:
                        no_impr += 1
                        stats["scheduler"] = "| no impr, best = {:.4f}".format(
                            self.scheduler.best)
                    else:
                        best_loss = cv["loss"]
                        no_impr = 0
                        self.save_checkpoint(best=True)
                    self.logger.info(
                        "{title} {tr} | {cv} {scheduler}".format(**stats))
                    # schedule here
                    self.scheduler.step(cv["loss"])
                    # flush scheduler info
                    sys.stdout.flush()
                    # save last checkpoint
                    self.save_checkpoint(best=False)
                    # reset reporter
                    train_reporter.reset()
                    # early stop or not
                    if no_impr == self.no_impr:
                        self.logger.info(
                            "Stop training cause no impr for {:d} epochs".
                            format(no_impr))
                        stop = True
                        break
                    if self.cur_epoch == num_epoches:
                        stop = True
                        break
                    # enable train mode
                    self.logger.info("Set train mode...")
                    self.nnet.train()
            self.logger.info("Finished one epoch on training set")
            if stop:
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epoches))
