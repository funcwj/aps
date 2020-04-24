#!/usr/bin/env python

import pathlib
import argparse

import torch as th
import numpy as np

from asr.utils import get_logger
from asr.eval import Computer
from asr.loader.am.wav import WaveReader

from kaldi_python_io import ScriptReader
from kaldi_python_io import Reader as BaseReader

logger = get_logger(__name__)


class MaskComputer(Computer):
    """
    Mask computation wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        super(MaskComputer, self).__init__(cpt_dir, device_id=device_id)
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def compute(self, wav, stats="mask"):
        wav = th.from_numpy(wav).to(self.device)[None, ...]
        if stats == "mask":
            out, _, _, _ = self.nnet.pred_mask(wav, None)
        else:
            out, _ = self.nnet.mvdr_beam(wav, None)
            out = out.abs()
        return out.detach().cpu().squeeze().numpy()


def run(args):
    computer = MaskComputer(args.checkpoint, device_id=args.device_id)
    wav_reader = WaveReader(args.wav_scp, sr=16000)

    dump_dir = pathlib.Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for key, wav in wav_reader:
        logger.info(f"Processing utterance {key}...")
        mask = computer.compute(wav, stats=args.stats)
        np.save(dump_dir / key, mask)
    logger.info(f"Processed {len(wav_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute attention alignments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint",
                        type=str,
                        help="Checkpoint of the multi-channel AM")
    parser.add_argument("wav_scp", type=str, help="Feature/Wave scripts")
    parser.add_argument("--stats",
                        type=str,
                        choices=["beam", "mask"],
                        help="Type of the output statistics")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="mask",
                        help="Output directory for TF-masks")
    args = parser.parse_args()
    run(args)