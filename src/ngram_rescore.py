#!/usr/bin/env python

# wujian@2019

import kenlm
import codecs
import argparse

from aps.nn.lm.utils import NbestReader
from aps.utils import get_logger, io_wrapper
from aps.utils import StrToBoolAction

logger = get_logger(__name__)


def run(args):
    nbest_reader = NbestReader(args.nbest)
    ngram = kenlm.Model(args.lm)

    stdout, top1 = io_wrapper(args.top1, "w")
    for key, nbest in nbest_reader:
        rescore = []
        for hyp in nbest:
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
    logger.info(f"Rescore {len(nbest_reader)} utterances " +
                f"on {nbest_reader.nbest} hypos")


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