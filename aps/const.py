# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Give pre-defined values
"""

import math
import numpy as np
import torch as th

IGNORE_ID = -1
NEG_INF = th.finfo(th.float32).min
MATH_PI = math.pi
EPSILON = np.finfo(np.float32).eps
MAX_INT16 = np.iinfo(np.int16).max
UNK_TOKEN = "<unk>"
