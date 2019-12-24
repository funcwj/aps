# wujian@2019

import os
import math
import pathlib

import torch as th
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .utils import load_obj
from .trainer import ProgressReporter, EarlyStopCriterion, IGNORE_ID
from .trainer import get_device_ids, add_gaussian_noise
from .trainer import ce_loss, ls_loss, compute_accu
from .scheduler import support_ss_scheduler


class Trainer(object):
    """
    A PyTorch base trainer
    """
    def __init__(self,
                 rank,
                 nnet,
                 cuda_devices=1,
                 dist_url="env://",
                 checkpoint="cpt",
                 optimizer="adam",
                 optimizer_kwargs=None,
                 lr_scheduler_kwargs=None,
                 lsm_factor=0,
                 ss_scheduler="const",
                 ss_prob=0,
                 ss_scheduler_kwargs=None,
                 grad_clip=None,
                 grad_noise=None,
                 prog_interval=100,
                 save_interval=-1,
                 resume="",
                 init="",
                 tensorboard=False,
                 stop_criterion="loss",
                 no_impr=6):
        if rank >= cuda_devices:
            raise ValueError("rank value exceeds number of GPUs: " +
                             f"{rank} vs {cuda_devices}")
        if dist_url == "env://":
            if int(os.environ.get("RANK")) != rank:
                raise RuntimeError(
                    f"rank value {rank} mismatch with env[RANK]")
            if int(os.environ.get("WORLD_SIZE")) != cuda_devices:
                raise RuntimeError(
                    f"world size {cuda_devices} mismatch with env[WORLD_SIZE]")

        self.eos = nnet.eos
        self.rank = rank
        self.cuda_devices = cuda_devices
        self.default_device = th.device(f"cuda:{rank:d}")

        self.checkpoint = pathlib.Path(checkpoint)
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        self.reporter = ProgressReporter(self.checkpoint,
                                         rank=rank,
                                         period=prog_interval,
                                         tensorboard=tensorboard)

        self.grad_clip = grad_clip
        self.grad_noise = grad_noise
        self.cur_epoch = 0  # zero based
        self.save_interval = save_interval
        self.lsm_factor = lsm_factor
        self.ssr = 0

        mode = "max" if stop_criterion == "accu" else "min"
        self.stop_criterion = stop_criterion
        self.early_stop = EarlyStopCriterion(no_impr, mode=mode)
        self.ss_scheduler = support_ss_scheduler(ss_scheduler, ss_prob,
                                                 **ss_scheduler_kwargs)

        self.reporter.log(f"Model summary:\n{nnet}")
        if resume or init:
            cpt_path = resume if resume else init
            if not pathlib.Path(cpt_path).exists():
                raise FileNotFoundError(
                    f"Could not find checkpoint: {cpt_path}")
            cpt = th.load(cpt_path, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            nnet.load_state_dict(cpt["model_state_dict"])
            self.nnet = self.setup_distributed(nnet, dist_url)
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
            self.nnet = self.setup_distributed(nnet, dist_url)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(self.optimizer,
                                           mode=mode,
                                           **lr_scheduler_kwargs)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.reporter.log(
            f"running process {rank} on GPU-{rank}/{self.cuda_devices}, " +
            f"#param: {self.num_params:.2f}M")
        self.reporter.log(f"Schedule sampling strategy: {ss_scheduler}")
        self.reporter.log(f"Early stop criterion: {stop_criterion}")
        if grad_clip:
            self.reporter.log(f"Gradient clipping by {grad_clip}, default L2")
        if grad_noise:
            self.reporter.log(
                f"Add gaussian noise to weights with std = {grad_noise}")

    def setup_distributed(self, nnet, dist_url):
        """
        Setup environment for distributed training
        """
        th.backends.cudnn.benchmark = True
        th.cuda.set_device(self.default_device)
        # offload to cuda device
        nnet.to(self.default_device)
        if self.cuda_devices >= 2:
            self.distributed = True
            # do distributed
            dist.init_process_group(backend="nccl",
                                    init_method=dist_url,
                                    rank=self.rank,
                                    world_size=self.cuda_devices)
            self.reporter.log(
                f"init process group, rank={self.rank}, " +
                f"world_size={self.cuda_devices}, init_method={dist_url}")
            return DistributedDataParallel(nnet, device_ids=[self.rank])
        else:
            self.distributed = False
            return nnet

    def save_checkpoint(self, epoch, best=True):
        """
        Save checkpoint (epoch, model, optimizer)
        """
        if self.rank == 0:
            cpt = {
                "epoch":
                epoch,
                "model_state_dict":
                self.nnet.module.state_dict()
                if self.distributed else self.nnet.state_dict(),
                "optim_state_dict":
                self.optimizer.state_dict()
            }
            cpt_name = "{}.pt.tar".format("best" if best else "last")
            th.save(cpt, self.checkpoint / cpt_name)
            self.reporter.log(f"Save checkpoint {cpt_name}")
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
            if self.grad_clip:
                norm = clip_grad_norm_(self.nnet.parameters(), self.grad_clip)

            loss = loss.item()
            if math.isfinite(norm) and math.isfinite(loss):
                self.optimizer.step()

                if self.grad_noise:
                    add_gaussian_noise(self.nnet, std=self.grad_noise)

                self.reporter.add("norm", norm)
                self.reporter.add("loss", loss)
                self.reporter.add("accu", accu)
            else:
                self.reporter.log(f"Invalid gradient {norm} or " +
                                  f"loss {loss}, skip...")

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

    def _prep_train(self, valid_loader):
        """
        Prepare for training
        """
        # avoid alloc memory from gpu0
        th.cuda.set_device(self.default_device)
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # eval
        self.eval(valid_loader)
        e = self.cur_epoch
        best_loss, best_accu, _ = self.reporter.report(e, 0)
        self.ssr = self.ss_scheduler.step(e, best_accu)
        # make sure not inf
        best_value = best_loss if self.stop_criterion == "loss" else best_accu
        self.scheduler.best = best_value
        self.early_stop.reset(best_value)
        # log here
        self.reporter.log(
            f"Epoch {e:d}, loss = {best_loss:.4f}, accu = {best_accu:.2f}")
        return e

    def run(self, train_loader, valid_loader, num_epoches=50):
        """
        Run on whole training set and evaluate
        """
        e = self._prep_train(valid_loader)
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

            update_value = cv_loss if self.stop_criterion == "loss" else cv_accu
            better = self.early_stop.step(update_value)
            if better:
                self.save_checkpoint(e, best=True)
            else:
                sstr += f" | no impr, best = {self.scheduler.best:.4f}"

            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(update_value)
            self.ssr = self.ss_scheduler.step(e, cv_accu)
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
        """
        Run on several batches and evaluate
        """
        e = self._prep_train(valid_loader)
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

                if self.grad_clip:
                    norm = clip_grad_norm_(self.nnet.parameters(),
                                           self.grad_clip)
                loss = loss.item()
                if math.isfinite(norm) and math.isfinite(loss):
                    self.optimizer.step()

                    if self.grad_noise:
                        add_gaussian_noise(self.nnet, std=self.grad_noise)

                    self.reporter.add("norm", norm)
                    self.reporter.add("loss", loss)
                    self.reporter.add("accu", accu)
                else:
                    self.reporter.log(f"Invalid gradient {norm} or " +
                                      f"loss {loss}, skip...")

                # if trained on batches done, start evaluation
                if trained_batches == 0:
                    e += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, _, sstr = self.reporter.report(e, cur_lr)
                    self.reporter.log(sstr)

                    cv_loss, cv_accu, sstr = self.reporter.report(e, cur_lr)
                    # schedule sampling for eval
                    sstr += f" | ssr = {self.ssr:.3f}"

                    update_value = cv_loss if self.stop_criterion == "loss" else cv_accu
                    better = self.early_stop.step(update_value)
                    if better:
                        self.save_checkpoint(e, best=True)
                    else:
                        sstr += f" | no impr, best = {self.scheduler.best:.4f}"

                    self.reporter.log(sstr)
                    # schedule here
                    self.scheduler.step(update_value)
                    self.ssr = self.ss_scheduler.step(e, cv_accu)
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
    def __init__(self, rank, nnet, ctc_coeff=0, ctc_blank=0, **kwargs):
        super(S2STrainer, self).__init__(rank, nnet, **kwargs)
        if ctc_coeff:
            self.reporter.log("Using CTC regularization (coeff = " +
                              f"{ctc_coeff:.2f}, blank = {ctc_blank})")
        self.ctc_coeff = ctc_coeff
        self.ctc_blank = ctc_blank

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
        tgt_pad = egs["tgt_pad"].masked_fill(ignored_mask, self.eos)
        # outs: N x (To+1) x V
        # alis: N x (To+1) x Ti
        outs, _, ctc_branch, enc_len = self.nnet(egs["src_pad"],
                                                 egs["src_len"], tgt_pad, ssr)
        # N x (To+1), pad -1
        tgts = F.pad(egs["tgt_pad"], (0, 1), value=IGNORE_ID)
        # add eos
        tgts = tgts.scatter(1, egs["tgt_len"][:, None], self.eos)
        # compute loss
        if self.lsm_factor > 0:
            loss = ls_loss(outs, tgts, lsm_factor=self.lsm_factor)
        else:
            loss = ce_loss(outs, tgts)

        if self.ctc_coeff > 0:
            # add log-softmax, N x T x V => T x N x V
            log_prob = F.log_softmax(ctc_branch, dim=-1).transpose(0, 1)
            # CTC loss
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