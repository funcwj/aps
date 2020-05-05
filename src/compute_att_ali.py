#!/usr/bin/env python

import argparse

import torch as th
import numpy as np

from aps.eval import Computer
from aps.utils import get_logger
from aps.loader.am.wav import WaveReader

from kaldi_python_io import ScriptReader
from kaldi_python_io import Reader as BaseReader

logger = get_logger(__name__)


class AlignmentComputer(Computer):
    """
    Alignments computation wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        super(AlignmentComputer, self).__init__(cpt_dir, device_id=device_id)
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def run(self, src, token):
        src = th.from_numpy(src).to(self.device)
        token = th.tensor(token, dtype=th.int64, device=self.device)
        prob, alis, _, _ = self.nnet(src[None, :], None, token[None, :], ssr=1)
        pred = th.argmax(prob.detach().squeeze(0), -1)[:-1]
        accu = th.sum(pred == token).float() / token.size(-1)
        logger.info(f"Accu = {accu.item():.2f}")
        return alis.detach().cpu().squeeze().numpy()


def run(args):
    token_reader = BaseReader(args.token,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1,
                              restrict=False)
    computer = AlignmentComputer(args.checkpoint, device_id=args.device_id)
    if computer.accept_raw:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    dump_dir = pathlib.Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("token",
                        type=str,
                        help="Rspecifier for evaluation transcriptions")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the acoustic model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="att_ali",
                        help="Output directory for alignments")
    args = parser.parse_args()
    run(args)