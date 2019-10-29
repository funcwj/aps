#!/usr/bin/env python

import os
import json
import argparse

import torch as th
import numpy as np

from libs.evaluator import Evaluator
from libs.logger import get_logger

from kaldi_python_io import ScriptReader
from kaldi_python_io import Reader as BaseReader

from seq2seq import Seq2Seq

logger = get_logger(__name__)


class Computer(Evaluator):
    """
    Computer wrapper
    """
    def __init__(self, *args, **kwargs):
        super(Computer, self).__init__(*args, **kwargs)

    def compute(self, feats, token):
        feats = th.from_numpy(feats).to(self.device)
        token = th.tensor(token, dtype=th.int64, device=self.device)
        prob, ali = self.nnet(feats[None, :], None, token[None, :], ssr=1)
        pred = th.argmax(prob.detach().squeeze(0), -1)[:-1]
        accu = th.sum(pred == token).float() / token.size(-1)
        logger.info("Accu = {:.2f}".format(accu.item()))
        return ali.detach().cpu().squeeze().numpy()


def run(args):

    os.makedirs(args.dump_dir, exist_ok=True)

    feats_reader = ScriptReader(args.feats_scp)
    token_reader = BaseReader(args.token_scp,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1)
    computer = Computer(Seq2Seq, args.checkpoint, gpu_id=args.gpu)
    for key, feats in feats_reader:
        logger.info("Processing utterance {}...".format(key))
        alis = computer.compute(feats, token_reader[key])
        np.save(os.path.join(args.dump_dir, key), alis)
    logger.info("Processed {:d} utterance done".format(len(feats_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute attention alignments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint",
                        type=str,
                        help="Checkpoint of the E2E model")
    parser.add_argument("feats_scp",
                        type=str,
                        help="Rspecifier for evaluation feature")
    parser.add_argument("token_scp",
                        type=str,
                        help="Rspecifier for evaluation transcriptions")
    parser.add_argument("--gpu",
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