#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import kenlm
import codecs
import argparse

from aps.utils import get_logger, io_wrapper
from aps.opts import StrToBoolAction
from typing import Dict, Tuple

logger = get_logger(__name__)


def read_nbest(nbest_fd: str) -> Tuple[int, Dict]:
    hypos = {}
    nbest = 1
    with codecs.open(nbest_fd, "r", encoding="utf-8") as f:
        nbest = int(f.readline())
        while True:
            key = f.readline().strip()
            if not key:
                break
            topk = []
            for _ in range(nbest):
                items = f.readline().strip().split()
                score = float(items[0])
                trans = " ".join(items[1:])
                topk.append((score, trans))
            hypos[key] = topk
    return (nbest, hypos)


def run(args):
    nbest, nbest_hypos = read_nbest(args.nbest)
    ngram = kenlm.Model(args.lm)

    stdout, top1 = io_wrapper(args.top1, "w")
    for key, nbest_dict in nbest_hypos.items():
        rescore = []
        for hyp in nbest_dict:
            am_score, trans = hyp
            lm_score = ngram.score(trans, bos=True, eos=True)
            if args.normalized:
                am_score /= len(trans)
            score = am_score + args.alpha * lm_score
            rescore.append((score, trans))
        rescore = sorted(rescore, key=lambda n: n[0], reverse=True)
        top1.write(f"{key}\t{rescore[0][1]}\n")
    if not stdout:
        top1.close()
    logger.info(f"Rescore {len(nbest_hypos)} utterances on {nbest} hypos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to rescore nbest hypothesis using ngram LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nbest", type=str, help="Nbest hypothesis")
    parser.add_argument("lm", type=str, help="Ngram LM")
    parser.add_argument("top1", type=str, help="Rescored best hypothesis")
    parser.add_argument("--alpha",
                        type=float,
                        default=0.1,
                        help="Language model weight")
    parser.add_argument("--normalized",
                        action=StrToBoolAction,
                        default="true",
                        help="If ture, using length normalized "
                        "for acoustic score")
    args = parser.parse_args()
    run(args)
