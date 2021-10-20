#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import sys
import codecs

from typing import Tuple, Any


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
