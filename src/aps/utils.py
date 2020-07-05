#!/usr/bin/env python

# wujian@2019

import sys
import time
import codecs
import argparse
import logging

import torch as th

__all__ = [
    "get_logger", "load_obj", "get_device_ids", "SimpleTimer",
    "StrToBoolAction"
]

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
        device_ids = (device_ids, )
    elif isinstance(device_ids, str):
        device_ids = tuple(map(int, device_ids.split(",")))
    else:
        raise ValueError(f"Unsupported value for device_ids: {device_ids}")
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


class StrToBoolAction(argparse.Action):
    """
    Since argparse.store_true is not very convenient
    """
    def __call__(self, parser, namespace, values, option_string=None):
        def str2bool(value):
            if value in ["true", "True"]:
                return True
            elif value in ["False", "false"]:
                return False
            else:
                raise ValueError

        try:
            setattr(namespace, self.dest, str2bool(values))
        except ValueError:
            raise Exception(f"Unknown value {values} for --{self.dest}")