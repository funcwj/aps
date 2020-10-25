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

default_logger_format = "%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s"


def get_logger(name,
               format_str=default_logger_format,
               date_format="%Y-%m-%d %H:%M:%S",
               file=False):
    """
    Get logger instance
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


def io_wrapper(io_str, mode):
    """
    Wrapper for IO stream
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


def set_seed(seed_str):
    """
    Set random seed for numpy & torch & cuda
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

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60
