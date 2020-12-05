#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Beam search for CTC
"""
import torch as th

from typing import List, Dict


def greedy_search(enc_out: th.Tensor, blank: int = 0) -> List[Dict]:
    """
    Greedy search
    """
    pass
