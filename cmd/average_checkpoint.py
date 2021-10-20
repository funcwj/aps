#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import torch as th

from aps.trainer.base import ParameterAverager
from aps.utils import get_logger

logger = get_logger(__name__)


def run(args):
    averager = ParameterAverager()
    num_parameters = -1
    for cpt in args.checkpoints:
        logger.info(f"Loading {cpt} ...")
        cpt = th.load(cpt, map_location="cpu")
        averager.add(cpt["model_state"])
        if num_parameters == -1:
            num_parameters = cpt["num_parameters"]
    avg = {
        "step": -1,
        "epoch": -1,
        "model_state": averager.state_dict(),
        "num_parameters": num_parameters
    }
    th.save(avg, args.output)
    logger.info(f"Average checkpoint ==> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to average checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoints",
                        nargs="+",
                        type=str,
                        help="The checkpoints used for average")
    parser.add_argument("output",
                        type=str,
                        help="Output path for averaged checkpoint")
    args = parser.parse_args()
    run(args)
