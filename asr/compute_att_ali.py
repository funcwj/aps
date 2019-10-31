#!/usr/bin/env python

import argparse

import torch as th
import numpy as np

from pathlib import Path

from libs.evaluator import Evaluator
from libs.utils import get_logger
from loader.wav_loader import WaveReader

from kaldi_python_io import ScriptReader
from kaldi_python_io import Reader as BaseReader

from seq2seq import Seq2Seq
from transform.asr import FeatureTransform

logger = get_logger(__name__)


class Computer(Evaluator):
    """
    Computer wrapper
    """
    def __init__(self, *args, **kwargs):
        super(Computer, self).__init__(*args, **kwargs)

    def compute(self, src, token):
        src = th.from_numpy(src).to(self.device)
        token = th.tensor(token, dtype=th.int64, device=self.device)
        prob, ali = self.nnet(src[None, :], None, token[None, :], ssr=1)
        pred = th.argmax(prob.detach().squeeze(0), -1)[:-1]
        accu = th.sum(pred == token).float() / token.size(-1)
        logger.info(f"Accu = {accu.item():.2f}")
        return ali.detach().cpu().squeeze().numpy()


def run(args):

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    token_reader = BaseReader(args.token_scp,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1)
    computer = Computer(Seq2Seq,
                        FeatureTransform,
                        args.checkpoint,
                        device_id=args.device_id)
    if computer.raw_waveform:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    for key, src in src_reader:
        logger.info(f"Processing utterance {key}...")
        alis = computer.compute(src, token_reader[key])
        np.save(dump_dir / key, alis)
    logger.info(f"Processed {len(src_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute attention alignments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("token_scp",
                        type=str,
                        help="Rspecifier for evaluation transcriptions")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the E2E model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="ali",
                        help="Output directory for alignments")
    args = parser.parse_args()
    run(args)