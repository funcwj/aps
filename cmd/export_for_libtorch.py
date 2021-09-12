#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import torch as th
import torch.nn as nn

from typing import Optional
from aps.eval.wrapper import NnetEvaluator
from aps.transform.utils import export_jit
from aps.utils import get_logger

logger = get_logger(__name__)


class ConcatTransform(nn.Module):
    """
    Concatenation of two transform results
    """

    def __init__(self, mag_transform: nn.Module,
                 ipd_transform: Optional[nn.Module]) -> None:
        super(ConcatTransform, self).__init__()
        self.mag_transform = mag_transform
        self.ipd_transform = ipd_transform

    def forward(self, packed: th.Tensor) -> th.Tensor:
        """
        Args:
            packed (Tensor): STFT results in real format, N x C x F x T x 2
        Return:
            feats (Tensor): spectral + spatial features, N x T x ...
        """
        feats = [self.mag_transform(packed)]
        if self.ipd_transform is not None:
            feats.append(self.ipd_transform(packed))
        return th.cat(feats, -1)


def scripted_and_save(nnet: nn.Module, path: str) -> None:
    """
    Scripted nnet and save it
    """
    scripted_nnet = th.jit.script(nnet)
    logger.info(scripted_nnet)
    th.jit.save(scripted_nnet, path)
    logger.info(f"Save scripted nnet to {path}")


def run(args):
    evaluator = NnetEvaluator(args.checkpoint, cpt_tag=args.tag, device_id=-1)
    nnet = evaluator.nnet
    if hasattr(nnet, "asr_transform"):
        # export ASR transform
        transform_export = export_jit(nnet.asr_transform)
        scripted_and_save(transform_export,
                          f"{args.checkpoint}/{args.tag}.transform.pt")
        nnet.asr_transform = None
    if hasattr(nnet, "enh_transform"):
        enh_transform = nnet.enh_transform
        if enh_transform.mag_transform is not None:
            transform = ConcatTransform(export_jit(enh_transform.mag_transform),
                                        enh_transform.ipd_transform)
            scripted_and_save(transform,
                              f"{args.checkpoint}/{args.tag}.transform.pt")
        nnet.enh_transform = None
    scripted_and_save(nnet, f"{args.checkpoint}/{args.tag}.nnet.pt")


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
