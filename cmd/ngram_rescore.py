#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import kenlm
import argparse

from aps.utils import get_logger
from aps.opts import StrToBoolAction
from aps.io import io_wrapper, NBestReader

logger = get_logger(__name__)


def run(args):
    nbest_reader = NBestReader(args.nbest)
    ngram = kenlm.LanguageModel(args.lm)

    stdout, top1 = io_wrapper(args.top1, "w")
    done = 0
    for key, nbest in nbest_reader:
        rescore = []
        for hyp in nbest:
            am_score, num_tokens, trans = hyp
            lm_score = ngram.score(trans, bos=True, eos=True)
            if args.len_norm:
                am_score /= num_tokens
            score = am_score + args.lm_weight * lm_score
            rescore.append((score, trans))
        rescore = sorted(rescore, key=lambda n: n[0], reverse=True)
        top1.write(f"{key}\t{rescore[0][1]}\n")
        done += 1
    if not stdout:
        top1.close()
    logger.info(f"Rescore {done} utterances on {nbest} hypos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to rescore nbest hypothesis using ngram LM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nbest", type=str, help="Nbest hypothesis")
    parser.add_argument("lm", type=str, help="Ngram LM")
    parser.add_argument("top1", type=str, help="Rescored best hypothesis")
    parser.add_argument("--lm-weight",
                        type=float,
                        default=0.1,
                        help="Language model weight")
    parser.add_argument("--len-norm",
                        action=StrToBoolAction,
                        default=True,
                        help="If ture, using length normalized "
                        "for acoustic score")
    args = parser.parse_args()
    run(args)
