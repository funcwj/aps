# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Give pre-defined values
"""

import math
import numpy as np
import torch as th

from distutils.version import LooseVersion

IGNORE_ID = -1
MIN_F32 = th.finfo(th.float32).min
NEG_INF = float("-inf")
MATH_PI = math.pi
EPSILON = float(np.finfo(np.float32).eps)
MAX_INT16 = np.iinfo(np.int16).max
UNK_TOKEN = "<unk>"
BLK_TOKEN = "<b>"
EOS_TOKEN = "<eos>"
SOS_TOKEN = "<sos>"
OOM_STRING = "out of memory"
TORCH_VERSION = LooseVersion(th.__version__)
