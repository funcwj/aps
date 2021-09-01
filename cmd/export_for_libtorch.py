#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import torch as th
from aps.eval.wrapper import NnetEvaluator
from aps.utils import get_logger

logger = get_logger(__name__)


def run(args):
    evaluator = NnetEvaluator(args.checkpoint, cpt_tag=args.tag, device_id=-1)
    nnet = evaluator.nnet
    if hasattr(nnet, "asr_transform"):
        nnet.asr_transform = None
    if hasattr(nnet, "enh_transform"):
        nnet.enh_transform = None
    scripted_nnet = th.jit.script(nnet)
    scripted_cpt = f"{args.checkpoint}/{args.tag}.scripted.pt"
    th.jit.save(scripted_nnet, scripted_cpt)
    logger.info(f"Save scripted nnet to {scripted_cpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to export out PyTorch's model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Path of the checkpoint")
    parser.add_argument("--tag",
                        type=str,
                        default="best",
                        help="Name of the model to export out")
    args = parser.parse_args()
    run(args)
