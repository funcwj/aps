#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from os import environ
from typing import NoReturn
import torch as th
import torch.distributed as dist

try:
    import horovod.torch as hvd
    hvd_available = True
except ImportError:
    hvd_available = False

BACKEND = "none"


def init(backend) -> NoReturn:
    """
    Set distributed backend
    """
    if backend not in ["torch", "horovod", "none"]:
        raise ValueError(f"Unsupported backend: {backend}")
    global BACKEND
    BACKEND = backend
    if backend == "horovod":
        hvd.init()
    if backend == "torch":
        for env in ["LOCAL_RANK", "WORLD_SIZE"]:
            if env not in environ:
                raise RuntimeError(
                    f"Not found in {env} environments, using python "
                    "-m torch.distributed.launch to launch the command")
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                rank=int(environ["LOCAL_RANK"]),
                                world_size=int(environ["WORLD_SIZE"]))


def check_backend() -> NoReturn:
    """
    Check validity of the backend
    """
    if BACKEND not in ["horovod", "torch"]:
        raise RuntimeError("distributed backend is not initialized")
    if BACKEND == "horovod" and not hvd_available:
        raise RuntimeError("horovod not installed!")


def get_backend() -> str:
    """
    Get distributed backend
    """
    return BACKEND


def rank() -> int:
    """
    Return rank id
    """
    check_backend()
    if BACKEND == "torch":
        return dist.get_rank()
    else:
        return hvd.local_rank()


def world_size() -> int:
    """
    Return world size
    """
    check_backend()
    if BACKEND == "torch":
        return dist.get_world_size()
    else:
        return hvd.size()


def all_reduce(tensor: th.Tensor) -> th.Tensor:
    """
    Return tensor after all reduce
    """
    check_backend()
    if BACKEND == "torch":
        # default: sum
        dist.all_reduce(tensor)
        return tensor / world_size()
    else:
        # default: avg
        return hvd.allreduce(tensor)
