#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import torch.nn as nn
import torch.nn.functional as tf

from typing import Tuple, Optional
from aps.transform.asr import TFTransposeTransform
from aps.asr.xfmr.encoder import TransformerEncoder


class Quantizer(nn.Module):

    def __init__(self):
        pass
