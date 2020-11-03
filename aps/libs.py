# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import importlib
import torch.nn as nn

from aps.transform import transform_cls
from aps.loader import loader_cls
from aps.task import task_cls
from aps.asr import asr_nnet_cls
from aps.sse import sse_nnet_cls

from typing import Any, Dict, Iterable


def dynamic_importlib(sstr: str) -> Any:
    """
    Import lib from the given string: e.g., toy_nnet.py:ToyNet
    """
    pypath, name = sstr.split(":")
    lib = importlib.import_module(pypath)
    if hasattr(lib, name):
        return getattr(lib, name)
    else:
        raise ImportError(f"Import {sstr} failed")


def aps_dataloader(fmt: str = "am_raw", **kwargs) -> Iterable[Dict]:
    """
    Return DataLoader class supported by aps
    """
    if fmt in loader_cls:
        loader_impl = loader_cls[fmt]
    elif ":" in fmt:
        loader_impl = dynamic_importlib(fmt)
    else:
        raise RuntimeError(f"Unsupported DataLoader type: {fmt}")
    return loader_impl(**kwargs)


def aps_task(task: str, nnet: nn.Module, **kwargs) -> nn.Module:
    """
    Return Task class supported by aps
    """
    if task in task_cls:
        task_impl = task_cls[task]
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
    return aps_specific_nnet(name, transform_cls)


def aps_asr_nnet(nnet: str) -> nn.Module:
    """
    Return ASR networks supported by aps
    """
    return aps_specific_nnet(nnet, asr_nnet_cls)


def aps_sse_nnet(nnet: str) -> nn.Module:
    """
    Return SSE networks supported by aps
    """
    return aps_specific_nnet(nnet, sse_nnet_cls)
