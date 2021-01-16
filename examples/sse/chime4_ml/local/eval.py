#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pathlib
import argparse

import torch as th
import numpy as np

from aps.loader import AudioReader
from aps.utils import get_logger, SimpleTimer
from aps.eval import NnetEvaluator
from aps.sse.unsuper.rnn import permu_aligner

logger = get_logger(__name__)


class Separator(NnetEvaluator):
    """
    Decoder wrapper
    """

    def __init__(self,
                 cpt_dir: str,
                 device_id: int = -1,
                 cpt_tag: str = "best") -> None:
        super(Separator, self).__init__(cpt_dir,
                                        cpt_tag=cpt_tag,
                                        device_id=device_id,
                                        task="enh")
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def run(self, src: np.ndarray) -> th.Tensor:
        """
        Args:
            src (Array): (C) x S
        Return:
            mask (Tensor): T x F
        """
        src = th.from_numpy(src).to(self.device)
        return self.nnet.infer(src)


def run(args):
    sep_dir = pathlib.Path(args.sep_dir)
    sep_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(args.checkpoint, device_id=args.device_id)
    mix_reader = AudioReader(args.wav_scp, sr=args.sr)

    for key, mix in mix_reader:
        timer = SimpleTimer()
        mask = separator.run(mix)
        if isinstance(mask, th.Tensor):
            mask = mask.cpu().numpy()
            mask = np.stack([mask, 1 - mask])
        else:
            mask = np.stack([m.cpu().numpy() for m in mask])
        mask = permu_aligner(mask)
        # save TF-mask
        np.save(sep_dir / f"{key}", mask)
        time_cost = timer.elapsed() * 60
        dur = mix.shape[-1] / args.sr
        logger.info(
            f"Processing utterance {key} done, RTF = {time_cost / dur:.2f}")
    logger.info(f"Processed {len(mix_reader)} utterances done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do evaluate unsupervised enhancement model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp",
                        type=str,
                        help="Mixture & Noisy input audio scripts")
    parser.add_argument("sep_dir",
                        type=str,
                        help="Directory to dump enhanced/separated output")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the separation/enhancement model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the source audio")
    args = parser.parse_args()
    run(args)
