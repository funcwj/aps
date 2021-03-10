# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import warnings
import importlib
import torch.nn as nn

from os.path import basename
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, List, Iterable
from argparse import Namespace
from aps import distributed


class Register(dict):
    """
    A decorator class (see function register)
    """

    def __init__(self, name: str) -> None:
        super(Register, self).__init__()
        self.name = name

    def register(self, alias: str):

        def add(alias, obj):
            if alias in self.keys():
                warnings.warn(f"{alias}: {obj} has already " +
                              f"been registered in {self.name}")
            self[alias] = obj
            return obj

        return lambda obj: add(alias, obj)


class Module(object):
    """
    Module class
    """

    def __init__(self, base: str, module: List[str]) -> None:
        self.base = base
        self.module = module

    def import_all(self):
        """
        Import all the modules
        """
        for sub_module in self.module:
            importlib.import_module(".".join([self.base, sub_module]))


class ApsRegisters(object):
    """
    Maintain registers for aps package
    """
    asr = Register("asr")
    sse = Register("sse")
    task = Register("task")
    loader = Register("loader")
    trainer = Register("trainer")
    transform = Register("transform")
    container = [asr, sse, task, loader, trainer, transform]


class ApsModules(object):
    """
    Maintain modules in aps package
    """
    asr_submodules = [
        "att", "enh_att", "transducers", "lm.rnn", "lm.xfmr", "filter.mvdr",
        "filter.conv", "filter.google"
    ]
    sse_submodules = [
        "toy", "unsuper.rnn", "enh.crn", "enh.phasen", "enh.dcunet",
        "bss.dccrn", "bss.dprnn", "bss.tasnet", "bss.xfmr", "bss.dense_unet"
    ]
    loader_submodules = [
        "am.kaldi", "am.raw", "se.chunk", "se.online", "lm.utt", "lm.bptt"
    ]
    asr = Module("aps.asr", asr_submodules)
    sse = Module("aps.sse", sse_submodules)
    task = Module("aps.task", ["asr", "sse", "unsuper"])
    loader = Module("aps.loader", loader_submodules)
    trainer = Module("aps.trainer", ["ddp", "hvd", "apex"])
    transform = Module("aps.transform", ["asr", "enh"])


def dynamic_importlib(sstr: str) -> Any:
    """
    Import lib from the given string: e.g., toy_nnet.py:ToyNet
    """
    path, cls_name = sstr.split(":")
    pkg_name = basename(path).split(".")[0]
    loader = SourceFileLoader(pkg_name, path)
    libs = loader.load_module(pkg_name)
    if hasattr(libs, cls_name):
        return getattr(libs, cls_name)
    else:
        raise ImportError(f"Import {sstr} failed")


def aps_dataloader(fmt: str = "am@raw", **kwargs) -> Iterable[Dict]:
    """
    Return DataLoader class supported by aps
    """
    ApsModules.loader.import_all()
    if fmt in ApsRegisters.loader:
        loader_impl = ApsRegisters.loader[fmt]
    elif ":" in fmt:
        loader_impl = dynamic_importlib(fmt)
    else:
        raise RuntimeError(f"Unsupported DataLoader type: {fmt}")
    return loader_impl(**kwargs)


def aps_task(task: str, nnet: nn.Module, **kwargs) -> nn.Module:
    """
    Return Task class supported by aps
    """
    ApsModules.task.import_all()
    if task in ApsRegisters.task:
        task_impl = ApsRegisters.task[task]
    elif ":" in task:
        task_impl = dynamic_importlib(task)
    else:
        raise RuntimeError(f"Unsupported task: {task}")
    return task_impl(nnet, **kwargs)


def aps_specific_nnet(nnet: str, nnet_cls: str) -> nn.Module:
    """
    Return neural networks supported by aps
    """
    if nnet in nnet_cls:
        nnet_impl = nnet_cls[nnet]
    elif ":" in nnet:
        nnet_impl = dynamic_importlib(nnet)
    else:
        raise RuntimeError(f"Unsupported nnet: {nnet}")
    return nnet_impl


def aps_transform(name: str) -> nn.Module:
    """
    Return Transform networks supported by aps
    """
    ApsModules.transform.import_all()
    return aps_specific_nnet(name, ApsRegisters.transform)


def aps_asr_nnet(nnet: str) -> nn.Module:
    """
    Return ASR networks supported by aps
    """
    ApsModules.asr.import_all()
    return aps_specific_nnet(nnet, ApsRegisters.asr)


def aps_sse_nnet(nnet: str) -> nn.Module:
    """
    Return SSE networks supported by aps
    """
    ApsModules.sse.import_all()
    return aps_specific_nnet(nnet, ApsRegisters.sse)


def aps_trainer(trainer: str, distributed: bool = False) -> Any:
    """
    Return aps Trainer class
    """
    if not distributed and trainer not in ["ddp", "apex"]:
        raise ValueError(
            f"Single-GPU training doesn't support {trainer} Trainer")
    ApsModules.trainer.import_all()
    if trainer not in ApsRegisters.trainer:
        raise ValueError(f"Unknown Trainer class: {trainer}")
    return ApsRegisters.trainer[trainer]


def start_trainer(trainer: str,
                  conf: Dict,
                  nnet: nn.Module,
                  args: Namespace,
                  reduction_tag: str = "none",
                  other_loader_conf: Dict = None) -> None:
    """
    Run the instance of the aps Trainer
    """
    is_distributed = args.distributed != "none"
    if is_distributed:
        # init torch/horovod backend
        distributed.init(args.distributed)
        rank = distributed.rank()
    else:
        rank = None

    task = aps_task(conf["task"], nnet, **conf["task_conf"])

    TrainerClass = aps_trainer(args.trainer, distributed=is_distributed)
    # construct trainer
    # torch.distributed.launch will provide
    # environment variables, and requires that you use init_method="env://".
    trainer = TrainerClass(task,
                           rank=rank,
                           device_ids=args.device_ids,
                           checkpoint=args.checkpoint,
                           resume=args.resume,
                           init=args.init,
                           save_interval=args.save_interval,
                           prog_interval=args.prog_interval,
                           tensorboard=args.tensorboard,
                           reduction_tag=reduction_tag,
                           **conf["trainer_conf"])
    # save cmd options
    if rank in [0, None]:
        conf["cmd_args"] = vars(args)
        with open(f"{args.checkpoint}/train.yaml", "w") as f:
            yaml.dump(conf, f)

    # check if #devices == world_size
    if is_distributed:
        num_process = len(args.device_ids.split(","))
        if num_process != distributed.world_size():
            raise RuntimeError(
                f"Number of process != world size: {num_process} " +
                f"vs {distributed.world_size()}")
    else:
        num_process = 1

    data_conf = conf["data_conf"]
    loader_conf = {
        "fmt": data_conf["fmt"],
        "num_workers": args.num_workers // num_process
    }
    loader_conf.update(data_conf["loader"])
    if other_loader_conf:
        loader_conf.update(other_loader_conf)

    trn_loader = aps_dataloader(train=True,
                                distributed=is_distributed,
                                max_batch_size=args.batch_size // num_process,
                                **loader_conf,
                                **data_conf["train"])
    dev_loader = aps_dataloader(train=False,
                                distributed=False,
                                max_batch_size=args.batch_size //
                                args.dev_batch_factor,
                                **loader_conf,
                                **data_conf["valid"])
    trainer.run(trn_loader,
                dev_loader,
                num_epochs=args.epochs,
                eval_interval=args.eval_interval)

    return trainer
