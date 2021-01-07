# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import warnings

from pathlib import Path
from collections import defaultdict, OrderedDict

import torch as th
from typing import Optional, Dict, List, Union, Tuple, NoReturn, Iterable
from aps.trainer.ss import SsScheduler
from aps.trainer.lr import LrScheduler
from aps.utils import load_obj, get_device_ids, get_logger, SimpleTimer
from aps.task import Task

try:
    from torch.utils.tensorboard import SummaryWriter
    tensorboard_available = True
except ImportError:
    tensorboard_available = False


class WeightNoiseAdder(object):
    """
    Add gaussian noise to the network weight
    Args:
        cfg: [beg, step, end], adding noise during
             training steps (beg, end), with a step size #step
        std: std of the gaussian noise
    """

    def __init__(self, cfg: List[int], std: float = 0.075) -> None:
        self.std = std
        self.beg, self.step, self.end = cfg

    def __call__(self, nnet: th.nn.Module, step: int) -> NoReturn:
        if step < self.beg:
            return
        if self.end > 0 and step > self.end:
            return
        if step - self.beg % self.step:
            return
        for p in nnet.parameters():
            if p.requires_grad:
                p.data += th.normal(0,
                                    self.std,
                                    size=p.data.shape,
                                    device=p.data.device)


class ProgressReporter(object):
    """
    A simple training progress reporter used in Trainer class
    Args:
        checkpoint: checkpoint directory (for logging file & tensorboard)
        metrics: matrics to track, e.g., accu, loss, @ppl
        rank: rank value (for distributed training only)
        tensorboard: use tensorboard or not
        reduction_tag: #utt|#tok|none, how we compute the averaged numbers
    """

    def __init__(self,
                 checkpoint: Path,
                 metrics: List[str],
                 rank: Optional[int] = None,
                 period: int = 100,
                 tensorboard: bool = True,
                 reduction_tag: str = "none") -> None:
        # NOTE on reduction_tag:
        #   1) for asr tasks we use #tok (token level)
        #   2) for sse tasks we use $utt (utterance level)
        self.rank = rank
        self.period = period
        self.reduction_tag = reduction_tag
        # mkdir
        checkpoint.mkdir(parents=True, exist_ok=True)
        if rank is None:
            logger_loc = (checkpoint / "trainer.log").as_posix()
            self.header = "Trainer"
        else:
            logger_loc = (checkpoint / f"trainer.rank.{rank}.log").as_posix()
            self.header = f"Rank {rank}"

        self.logger = get_logger(logger_loc, file=True)
        # only for rank-0
        if tensorboard and rank in [0, None]:
            if not tensorboard_available:
                warnings.warn("tensorboard not installed thus disable it...")
                self.board_writer = None
            else:
                self.board_writer = SummaryWriter(checkpoint)
        else:
            self.board_writer = None
        self.metrics = metrics
        self.reset()

    def log(self, sstr: str) -> NoReturn:
        """
        Log messages
        """
        self.logger.info(f"{self.header} - {sstr}")

    def eval(self) -> NoReturn:
        """
        Reset to eval mode
        """
        self.log(">> Set eval mode ...")
        self.mode = "valid"
        self.reset()

    def train(self) -> NoReturn:
        """
        Reset to training mode
        """
        self.log(">> Set train mode ...")
        self.mode = "train"
        self.reset()

    def reset(self) -> NoReturn:
        """
        Clear the status
        """
        self.stats = defaultdict(list)
        self.timer = SimpleTimer()

    def update(self,
               dict_obj: Dict,
               keys: Optional[List[str]] = None) -> NoReturn:
        """
        Track the recording items (multiple)
        """
        if dict_obj is None:
            return
        if keys is None:
            for key, value in dict_obj.items():
                if isinstance(value, th.Tensor):
                    value = value.item()
                self.add(key, value)
        else:
            for key in keys:
                if key in dict_obj:
                    self.add(key, dict_obj[key])

    def add(self, key: str, value: float) -> NoReturn:
        """
        Track one recording item
        """
        self.stats[key].append(value)
        N = len(self.stats[key])
        if not N % self.period:
            if key == "rate":
                # current learning rate
                cur = self.stats[key][-1]
                self.log(f"Processed {N:.2e} batches ({key} = {cur:.3e}) ...")
            elif key[0] == "#":
                # averged token/utterance numbers in the past
                cur = sum(self.stats[key][-self.period:]) // self.period
                self.log(f"Processed {N:.2e} batches ({key} = {cur:d}) ...")
            else:
                avg = self._report_metric(key, period=self.period)
                self.log(f"Processed {N:.2e} batches ({key} = {avg:+.2f}) ...")

    def _report_metric(self, key: str, period: int = 0):
        """
        Return the averaged tracked metric
        """
        nors = self.stats[key][-period:]
        # try to get denominator
        if self.reduction_tag in self.stats:
            dens = self.stats[self.reduction_tag][-period:]
        else:
            dens = None
            warnings.warn(f"{self.reduction_tag} not found in the tracked " +
                          "statistics, using simple average")
        if dens is None:
            # simple average
            avg = sum(nors) / len(nors)
        else:
            # weight sum and average
            avg = sum(nors[i] * d for i, d in enumerate(dens)) / sum(dens)
        if key == "accu":
            avg *= 100
        if key == "@ppl":
            avg = math.exp(avg)
        return avg

    def _report_metrics(self):
        """
        Report the tracked metrics (used for logging & scheduling)
        """
        reports = {}
        for metric in self.metrics:
            if metric not in self.stats:
                raise RuntimeError(
                    f"Metric {metric} is not tracked by the reporter")
            reports[metric] = self._report_metric(metric)
        return reports

    def report(self, epoch: int, lr: float) -> Tuple[Dict, str]:
        """
        Return the reports and log messages
        """
        N = len(self.stats["loss"])
        if self.mode == "valid":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"Loss on {N:d} batches: {sstr}")

        if N == 0:
            raise RuntimeError("No statistics to report")
        # Got reports
        reports = self._report_metrics()
        # Write tensorboard if needed
        if self.board_writer:
            for name, value in reports.items():
                self.board_writer.add_scalar(f"stats/{self.mode}", name, value)
        cost = self.timer.elapsed()

        header = "/".join(self.metrics)
        values = "/".join([f"{reports[metric]:.4f}" for metric in self.metrics])
        logstr = (f"Epoch {epoch:02d}/{self.mode}: {header}(time/#batch, " +
                  f"lr={lr:.3e}) = {values}({cost:.2f}m/{N:d})")
        return reports, logstr


class ErrorDetector(object):
    """
    Detect the training errors
    Args:
        stop_on_errors: maximum number of errors can exist
    """

    def __init__(self, stop_on_errors: int) -> None:
        self.stop_on_errors = stop_on_errors
        self.reset()

    def reset(self) -> NoReturn:
        """
        Reset status
        """
        self.counter = 0
        self.last_error_step = 0
        self.local_step = 0

    def stop(self) -> bool:
        """
        Stop training or not
        """
        return self.counter == self.stop_on_errors

    def step(self, status):
        """
        Make one step
        """
        self.local_step += 1
        if self.local_step - self.last_error_step == 1 and not status:
            self.counter += 1
            self.last_error_step = self.local_step
        else:
            self.counter = 0
        return self.stop()


class StopDetector(object):
    """
    To manage the early stop of the training
    Args:
        no_impr: int, maximum number of epochs that no improvement exists
        mode: min|max
        init_criterion: initial value for best criterion
        no_impr_thres: threshold to check whether there is improvement
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
        """
        Reset the best criterion number
        """
        self.best_criterion = update_value

    def stop(self) -> bool:
        """
        Stop training or not
        """
        return self.no_impr == self.max_no_impr

    @property
    def best(self) -> float:
        """
        Return the tracked best criterion number
        """
        return self.best_criterion

    def state_dict(self) -> Dict:
        """
        Return the state of the detector
        """
        return {key: value for key, value in self.__dict__.items()}

    def load_state_dict(self, state_dict: Dict) -> NoReturn:
        """
        Load the detector state.
        """
        self.__dict__.update(state_dict)

    def step(self, update_value: float) -> bool:
        """
        Make one step
        """
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
    The base trainer (to be inherited)
    Args:
        task: Task class from aps.task
        rank: rank value (for distributed training)
        device_ids: GPU device ID
        checkpoint: directory for checkpoint storage
        optimizer: optimizer name (see function create_optimizer)
        optimizer_kwargs: parameters for the optimizer
        lr_scheduler: name of the learning rate scheduler (see aps.trainer.lr)
        lr_scheduler_kwargs: parameters for the learning rate scheduler
        lr_scheduler_period: epoch|step, run lr_scheduler per-epoch or per-step
        ss_scheduler: schedule sampling strategy (see aps.trainer.ss)
        ss_scheduler_kwargs: parameters for the ss_scheduler
        clip_gradient: value of L2 norm for gradient clipping
        acmu_gradient: do gradient accumulation over #acmu_gradient mini-batches
        prog_interval: interval to log training progress
        save_interval: interval to save checkpoint
        resume: checkpoint to resume training
        init: checkpoint for model initialization
        tensorboard: use tensorboard or not
        no_impr: stop training when it reaches the number of epochs that no improvements exist
        average_checkpoint: average the checkpoints over no improvement epochs or not
        stop_criterion: do early stopping detection on which metrics (must in in report_metrics)
        report_metrics: metrics to be tracked during training
        reduction_tag: used in ProgressReporter
        stop_on_errors: stop training if #stop_on_errors consecutive errors exist
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
        if not isinstance(task, Task):
            raise TypeError(
                f"Trainer accepts Task object, but got {type(task)}")
        if lr_scheduler_period not in ["epoch", "step"]:
            raise ValueError(
                f"Unsupported lr_scheduler_period: {lr_scheduler_period}")
        if stop_criterion not in report_metrics:
            raise ValueError("stop_criterion is not included in " +
                             f"report_metrics: {stop_criterion}")
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

        self.reporter = ProgressReporter(self.checkpoint,
                                         report_metrics,
                                         rank=rank,
                                         period=prog_interval,
                                         tensorboard=tensorboard,
                                         reduction_tag=reduction_tag)
        if weight_noise_std is None:
            self.weight_noise_adder = None
        else:
            self.weight_noise_adder = WeightNoiseAdder(weight_noise_cfg,
                                                       std=weight_noise_std)

        self.clip_gradient = clip_gradient
        self.acmu_gradient = acmu_gradient
        self.cur_epoch = 0  # zero based
        self.cur_step = 0
        self.save_interval = save_interval
        self.ssr = 0
        self.no_impr = no_impr
        self.average_checkpoint = average_checkpoint

        mode = "max" if stop_criterion == "accu" else "min"
        self.stop_on = stop_criterion
        self.stop_detector = StopDetector(no_impr,
                                          mode=mode,
                                          no_impr_thres=no_impr_thres)
        self.stop_on_errors = stop_on_errors
        self.detector = ErrorDetector(stop_on_errors)
        self.task = task
        if self.rank in [0, None]:
            self.reporter.log(f"Model summary:\n{task.nnet}")
        self.task.to(self.default_device)

        _lr_scheduler_kwargs = lr_scheduler_kwargs.copy()
        if resume or init:
            self.cpt_stats, optimizer_dict = self.load_checkpoint(
                resume if resume else init, "resume" if resume else "init")
            _lr_scheduler_kwargs["state"] = self.cpt_stats["lr_scheduler_state"]
        else:
            self.cpt_stats, optimizer_dict = None, None
            _lr_scheduler_kwargs["state"] = None
        # make optimizer
        self.optimizer = self.create_optimizer(optimizer,
                                               optimizer_kwargs,
                                               state=optimizer_dict)
        self.optimizer.zero_grad()

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
            _lr_scheduler_kwargs.update(reduce_lr_kwargs)
        self.lr_scheduler = self.create_scheduler(lr_scheduler, self.optimizer,
                                                  **_lr_scheduler_kwargs)
        self.lr_scheduler_period = lr_scheduler_period

        # make ss scheduler
        if ss_scheduler_kwargs:
            if ss_scheduler not in SsScheduler:
                raise ValueError(f"Unsupported ss scheduler: {ss_scheduler}")
            if "accu" not in report_metrics:
                raise ValueError("When using schedule sampling, accu need to "
                                 "be tracked in report_metrics")
            self.ss_scheduler = SsScheduler[ss_scheduler](**ss_scheduler_kwargs)
            self.reporter.log(f"Using schedule sampling: {ss_scheduler}")
        else:
            self.ss_scheduler = None

        self.num_params = sum(
            [param.nelement() for param in task.nnet.parameters()]) / 10.0**6
        # logging
        if rank is None:
            self.reporter.log(f"Load model to GPU:{device_ids[0]}, " +
                              f"#param: {self.num_params:.2f}M")
        else:
            self.reporter.log(
                f"Load model to GPU-{rank}/{self.cuda_devices}, " +
                f"#param: {self.num_params:.2f}M")

        self.reporter.log(
            f"Track the metrics during training: {report_metrics}, " +
            f"reduction = {reduction_tag}")
        self.reporter.log(f"Early stop detected on metric: {self.stop_on}")
        if clip_gradient:
            self.reporter.log(f"Clip gradient if over {clip_gradient} L2 norm")
        if acmu_gradient > 1:
            self.reporter.log(
                f"Accumulate gradient per {acmu_gradient} batches")
        if weight_noise_std:
            self.reporter.log("Add gaussian noise to gradient, with " +
                              f"std = {weight_noise_std}")
        if save_interval > 0:
            self.reporter.log("Will save model states only in #epoch.pt.tar " +
                              f"(interval = {save_interval})")

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
        optim = supported_optimizer[optimizer](self.task.parameters(), **kwargs)
        self.reporter.log(f"Create optimizer {optimizer}: {kwargs}")
        if state is not None:
            optim.load_state_dict(state)
            self.reporter.log("Load optimizer state from the checkpoint")
        return optim

    def create_scheduler(self,
                         scheduler: str,
                         optimizer: th.optim.Optimizer,
                         state: Optional[Dict] = None,
                         **kwargs):
        """
        Return a learning rate scheduler
        """
        if scheduler not in LrScheduler:
            raise ValueError(f"Unsupported lr scheduler: {scheduler}")
        lr_scheduler = LrScheduler[scheduler](optimizer, **kwargs)
        self.reporter.log(f"Create scheduler {scheduler}: {kwargs}")
        if state is not None:
            lr_scheduler.load_state_dict(state)
            self.reporter.log("Load scheduler state from the checkpoint")
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
        self.task.nnet.load_state_dict(cpt_stats["model_state"])
        cpt_str = (f"checkpoint {cpt_path}: " +
                   f"epoch/step {cpt_stats['epoch']}/{cpt_stats['step']}")
        optimizer_dict = None
        if manner == "resume":
            self.reporter.log(f"Resume from {cpt_str}")
            optimizer_dict = cpt_stats["optimizer_state"]
            self.stop_detector.load_state_dict(cpt_stats["detector_state"])
            # set current epoch/step number
            self.cur_epoch = cpt_stats["epoch"]
            self.cur_step = cpt_stats["step"]
        else:
            self.reporter.log(f"Intialize from {cpt_str}")
        self.reporter.log(
            f"Loss tracked in the checkpoint: {cpt_stats['loss']:.3f}")
        return cpt_stats, optimizer_dict

    def model_states(self) -> Dict:
        """
        Return model states which will be saved in the checkpoint
        """
        raise NotImplementedError

    def save_checkpoint(self,
                        states: Dict,
                        tag: str = "best",
                        enable_subroutine: bool = True,
                        keep_optimizer: bool = True) -> NoReturn:
        """
        Save checkpoint (epoch, model, optimizer, ...)
        """
        if self.rank in [0, None]:
            if enable_subroutine:
                cpt = self.model_states()
                cpt.update(states)
            else:
                cpt = states
            cpt_name = f"{tag}.pt.tar"
            if not keep_optimizer and "optimizer_state" in cpt:
                _ = cpt.pop("optimizer_state")
            th.save(cpt, self.checkpoint / cpt_name)
            self.reporter.log(
                f"Save checkpoint ==> {self.checkpoint / cpt_name}")

    def average_checkpoints(self) -> NoReturn:
        """
        Average checkpoint over no improvement epochs
        """
        if not self.average_checkpoint:
            return
        if self.rank not in [0, None]:
            return
        self.reporter.log("Average checkpoints best.pt.tar + no_impr" +
                          f".(1..{self.no_impr}).pt.tar ...")
        averaged = OrderedDict()
        for i in range(self.no_impr + 1):
            name = f"no_impr.{i}.pt.tar" if i else "best.pt.tar"
            cpt = th.load(self.checkpoint / name, map_location="cpu")
            param = cpt["model_state"]
            for key in param.keys():
                p = param[key]
                if key not in averaged:
                    averaged[key] = p.clone()
                else:
                    averaged[key] += p
        for key in averaged:
            if averaged[key].is_floating_point():
                averaged[key].div_(self.no_impr + 1)
            else:
                averaged[key] //= (self.no_impr + 1)

        final = {
            "step": self.cur_step,
            "epoch": self.cur_epoch,
            "model_state": averaged,
            "num_parameters": self.num_params
        }
        self.save_checkpoint(final, tag="avg", enable_subroutine=False)

    def train_one_step(self, egs: Dict) -> bool:
        """
        Make one training step (return true if no error exists)

        1) Forward & Backword
        2) Clip Gradient
        3) Step optimizer
        4) Zero optimizer
        """
        raise NotImplementedError

    def lr_scheduler_step(self,
                          update_value: Optional[float],
                          end_at: str = "epoch") -> NoReturn:
        """
        Make one step in lr scheduler
        """
        if end_at == "step" and self.lr_scheduler_period == "step":
            self.lr_scheduler.step()
        if end_at == "epoch" and self.lr_scheduler_period == "epoch":
            if isinstance(self.lr_scheduler, LrScheduler["reduce_lr"]):
                self.lr_scheduler.step(update_value)
            else:
                self.lr_scheduler.step()

    def train_epoch(self, data_loader: Iterable[Dict]) -> bool:
        """
        Run one training epoch
        """
        self.task.train()
        self.reporter.train()
        self.detector.reset()
        for egs in data_loader:
            # load to gpu
            egs = self.prep_egs(egs)
            # make one training step
            succ = self.train_one_step(egs)
            if succ:
                self.cur_step += 1
            if self.detector.step(succ):
                break
        stop = self.detector.stop()
        if stop:
            self.reporter.log("Stop training as detecting " +
                              f"{self.stop_on_errors} consecutive errors")
        return (not stop)

    def valid_epoch(self, data_loader: Iterable[Dict]) -> NoReturn:
        """
        Run one validation epoch
        """
        self.task.eval()
        self.reporter.eval()

        with th.no_grad():
            for egs in data_loader:
                # load to gpu
                egs = self.prep_egs(egs)
                stats = self.task(egs)
                # update statistics
                self.reporter.update(egs, ["#utt", "#tok"])
                self.reporter.update(stats)

    def stop_detect(self, dev_loader: Iterable[Dict], lr: float) -> bool:
        """
        Run valid epoch and schedule training progress:

        1) schedule learning/sampling rate
        2) save checkpoint
        3) early stop detection
        """
        self.valid_epoch(dev_loader)
        reports, logstr = self.reporter.report(self.cur_epoch, lr)
        # schedule sampling for eval
        if self.ss_scheduler:
            logstr += f" | ssr = {self.ssr:.3f}"

        update_value = reports[self.stop_on]
        better = self.stop_detector.step(update_value)

        status = {
            "step": self.cur_step,
            "epoch": self.cur_epoch,
            "num_parameters": self.num_params,
            "detector_state": self.stop_detector.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict()
        }
        status.update(reports)
        if better:
            # save best checkpoint
            self.save_checkpoint(status, tag="best")
        else:
            no_impr = self.stop_detector.no_impr
            logstr += f" | no impr: {no_impr:d}, "
            logstr += f"best = {self.stop_detector.best:.4f}"
            # save for average
            if self.average_checkpoint:
                self.save_checkpoint(status,
                                     tag=f"no_impr.{no_impr:d}",
                                     keep_optimizer=False)

        self.reporter.log(logstr)
        # << valid
        # lr schedule here
        self.lr_scheduler_step(update_value, end_at="epoch")
        if self.ss_scheduler:
            self.ssr = self.ss_scheduler.step(self.cur_epoch, reports["accu"])
        # save last checkpoint
        self.save_checkpoint(status, tag="last")
        if self.save_interval > 0 and self.cur_epoch % self.save_interval == 0:
            self.save_checkpoint(status,
                                 tag=f"{self.cur_epoch}",
                                 keep_optimizer=False)
        # early stop
        if self.stop_detector.stop():
            self.reporter.log("Stop training cause no impr for " +
                              f"{self.no_impr} epochs")
            return True
        return False

    def prep_egs(self, egs: Dict) -> Dict:
        """
        Prepare training egs
        """
        egs = load_obj(egs, self.default_device)
        # use ssr = 0 when in eval mode
        if self.ss_scheduler:
            egs["ssr"] = self.ssr if self.task.training else 0
        return egs

    def prep_run(self, dev_loader: Iterable[Dict]) -> int:
        """
        Prepare for training
        """
        # valid
        self.valid_epoch(dev_loader)
        # log lr as 0
        reports, logstr = self.reporter.report(self.cur_epoch, 0)
        self.reporter.log(logstr)
        if self.ss_scheduler:
            self.ssr = self.ss_scheduler.step(self.cur_epoch, reports["accu"])
        # make sure not inf
        best = reports[self.stop_on]
        self.stop_detector.reset(best)
        # for ReduceLROnPlateau
        if hasattr(self.lr_scheduler, "best"):
            self.lr_scheduler.best = best

    def run_in_epoch(self,
                     trn_loader: Iterable[Dict],
                     dev_loader: Iterable[Dict],
                     num_epochs: int = 50) -> int:
        """
        Running in epoch mode: treat whole training set as one training epoch
        """
        while self.cur_epoch < num_epochs:
            trn_loader.set_epoch(self.cur_epoch)
            self.cur_epoch += 1
            # >> train
            if not self.train_epoch(trn_loader):
                break
            cur_lr = self.optimizer.param_groups[0]["lr"]
            _, logstr = self.reporter.report(self.cur_epoch, cur_lr)
            self.reporter.log(logstr)
            # << train
            if self.stop_detect(dev_loader, cur_lr):
                break
        return self.cur_epoch

    def run_in_batch(self,
                     trn_loader: Iterable[Dict],
                     dev_loader: Iterable[Dict],
                     num_epochs: int = 100,
                     eval_interval: int = 3000) -> int:
        """
        Running in batch mode: for large training set, treat several batches as one training epoch
        """
        stop = False
        while True:
            # trained on several batches
            for egs in trn_loader:
                # enable train mode
                if self.cur_step % eval_interval == 0:
                    self.task.train()
                    self.reporter.train()
                    self.detector.reset()
                    trn_loader.set_epoch(self.cur_epoch)
                # update per-batch
                egs = self.prep_egs(egs)
                succ = self.train_one_step(egs)
                if succ:
                    self.cur_step += 1
                if self.detector.step(succ):
                    self.reporter.log(
                        f"Stop training as detecting {self.stop_on_errors} " +
                        "consecutive errors")
                    stop = True
                    break
                # if trained on batches done, start evaluation
                if self.cur_step % eval_interval == 0 and succ:
                    self.cur_epoch += 1
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    _, logstr = self.reporter.report(self.cur_epoch, cur_lr)
                    self.reporter.log(logstr)
                    end = self.stop_detect(dev_loader, cur_lr)
                    if end or self.cur_epoch == num_epochs:
                        stop = True
                        break
            if stop:
                break
            self.reporter.log("Finished one epoch on training set")
        return self.cur_epoch

    def run(self,
            trn_loader: Iterable[Dict],
            dev_loader: Iterable[Dict],
            num_epochs: int = 100,
            eval_interval: int = -1) -> NoReturn:
        """
        Entry of the Trainer class
        """
        trn_batches = len(trn_loader) if len(trn_loader) else "unknown"
        dev_batches = len(dev_loader) if len(dev_loader) else "unknown"
        self.reporter.log(
            f"Number of batches (train/valid): {trn_batches}/{dev_batches}")
        self.prep_run(dev_loader)
        if eval_interval > 0:
            done_epoch = self.run_in_batch(trn_loader,
                                           dev_loader,
                                           num_epochs=num_epochs,
                                           eval_interval=eval_interval *
                                           self.acmu_gradient)
        else:
            done_epoch = self.run_in_epoch(trn_loader,
                                           dev_loader,
                                           num_epochs=num_epochs)
        self.average_checkpoints()
        self.reporter.log(
            f"Training for {done_epoch:d}/{num_epochs:d} epochs done!")
