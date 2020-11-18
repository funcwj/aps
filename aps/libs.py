# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import importlib
import torch.nn as nn

from os.path import basename
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Iterable


class Register(dict):
    """
    Register class
    """

    def __init__(self, name):
        super(Register, self).__init__()
        self.name = name

    def register(self, alias):

        def add(alias, obj):
            if alias in self.keys():
                warnings.warn(f"{alias}: {obj} has already " +
                              f"been registered in {self.name}")
            self[alias] = obj
            return obj

        return lambda obj: add(alias, obj)


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


def aps_dataloader(fmt: str = "am_raw", **kwargs) -> Iterable[Dict]:
    """
    Return DataLoader class supported by aps
    """
    importlib.import_module("loader", package="aps")
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
    importlib.import_module("task", package="aps")
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
    return aps_specific_nnet(name, ApsRegisters.transform)


def aps_asr_nnet(nnet: str) -> nn.Module:
    """
    Return ASR networks supported by aps
    """
    importlib.import_module("asr", package="aps")
    return aps_specific_nnet(nnet, ApsRegisters.asr)


def aps_sse_nnet(nnet: str) -> nn.Module:
    """
    Return SSE networks supported by aps
    """
    importlib.import_module("sse", package="aps")
    return aps_specific_nnet(nnet, ApsRegisters.sse)


def aps_trainer(trainer: str, distributed: bool = False) -> Any:
    """
    Return aps Trainer class
    """
    if not distributed and trainer not in ["ddp", "apex"]:
        raise ValueError(
            f"Single-GPU training doesn't support {trainer} Trainer")
    importlib.import_module("trainer", package="aps")
    if trainer not in ApsRegisters.trainer:
        raise ValueError(f"Unknown Trainer class: {trainer}")
    return ApsRegisters.trainer[trainer]
