#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import time
import random
import codecs
import logging

import torch as th
import numpy as np

from typing import NoReturn, Tuple, Any, Union, Optional

aps_logger_format = ("%(asctime)s [%(pathname)s:%(lineno)s - " +
                     "%(levelname)s ] %(message)s")
aps_time_format = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str,
               format_str: str = aps_logger_format,
               date_format: str = aps_time_format,
               file: bool = False) -> logging.Logger:
    """
    Get logger instance
    Args:
        name: logger name
        format_str|date_format: to configure logging format
        file: if true, treat name as the name of the logging file
    """

    def get_handler(handler):
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
        handler.setFormatter(formatter)
        return handler

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    # both stdout & file
    if file:
        logger.addHandler(get_handler(logging.FileHandler(name)))
    return logger


def io_wrapper(io_str: str, mode: str) -> Tuple[bool, Any]:
    """
    Wrapper for IO stream
    Args:
        io_str: "-" or file name
        mode: IO mode
    """
    if io_str != "-":
        std = False
        stream = codecs.open(io_str, mode, encoding="utf-8")
    else:
        std = True
        if mode not in ["r", "w"]:
            raise RuntimeError(f"Unknown IO mode: {mode}")
        if mode == "w":
            stream = codecs.getwriter("utf-8")(sys.stdout.buffer)
        else:
            stream = codecs.getreader("utf-8")(sys.stdin.buffer)
    return std, stream


def load_obj(obj: Any, device: Union[th.device, str]) -> Any:
    """
    Offload tensor object in obj to cuda device
    Args:
        obj: Arbitrary object
        device: target device ("cpu", "cuda" or th.device object)
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def get_device_ids(device_ids: Union[str, int]) -> Tuple[int]:
    """
    Got device ids
    Args:
        device_ids: int or string like "0,1"
    """
    if not th.cuda.is_available():
        raise RuntimeError("CUDA device unavailable... exist")
    # None or 0
    if not device_ids:
        # detect number of device available
        dev_cnt = th.cuda.device_count()
        device_ids = tuple(range(0, dev_cnt))
    elif isinstance(device_ids, int):
        device_ids = (device_ids,)
    elif isinstance(device_ids, str):
        device_ids = tuple(map(int, device_ids.split(",")))
    else:
        raise ValueError(f"Unsupported value for device_ids: {device_ids}")
    return device_ids


def set_seed(seed_str: str) -> Optional[int]:
    """
    Set random seed for numpy & torch & cuda
    Args:
        seed_str: string
    """
    # set random seed
    if not seed_str or seed_str == "none":
        return None
    else:
        seed = int(seed_str)
        random.seed(seed)
        np.random.seed(seed)
        th.random.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        return seed


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> NoReturn:
        self.start = time.time()

    def elapsed(self) -> float:
        return (time.time() - self.start) / 60
