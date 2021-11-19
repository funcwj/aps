#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

from pathlib import Path
from aps.asr.lm.ngram import NgramLM
from aps.utils import get_logger
from aps.eval import NnetEvaluator, TextPreProcessor
from aps.opts import StrToBoolAction
from aps.io import io_wrapper, NbestReader

logger = get_logger(__name__)


def run(args):
    nbest_reader = NbestReader(args.nbest)

    kenlm = Path(args.lm).is_file()
    tokenizer = None
    sos, eos = -1, -1
    if kenlm:
        lm = NgramLM(args.lm, args.dict)
        logger.info(f"Load ngram LM from {args.lm}, weight = {args.lm_weight}")
    else:
        lm = NnetEvaluator(args.lm, device_id=-1, cpt_tag=args.lm_tag)
        logger.info(f"Use NN LM weight: {args.lm_weight}")
        lm = lm.nnet
        tokenizer = TextPreProcessor(args.dict, space=args.space,
                                     spm=args.spm).tokenizer
        sos = tokenizer.symbol2int("<sos>")
        eos = tokenizer.symbol2int("<eos>")

    stdout, top1 = io_wrapper(args.top1, "w")
    done = 0
    for key, nbest in nbest_reader:
        rescore = []
        for hyp in nbest:
            # NOTE: am_score: without length normalization
            am_score, num_tokens, hypos = hyp
            if kenlm:
                lm_score = lm.score(hypos, eos=True, sos=True)
                if args.len_norm:
                    am_score /= num_tokens
                score = am_score + args.lm_weight * lm_score
            else:
                hypos_str_seq = hypos.split(" ")
                hypos_int_seq = tokenizer.encode(hypos_str_seq)
                lm_score = lm.score(hypos_int_seq, eos=eos, sos=sos)
                score = (am_score + lm_score) / num_tokens
            rescore.append((score, hypos))
        rescore = sorted(rescore, key=lambda n: n[0], reverse=True)
        top1.write(f"{key}\t{rescore[0][1]}\n")
        done += 1
    if not stdout:
        top1.close()
    logger.info(f"Rescore {done} utterances on {nbest} hypos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to rescore nbest hypothesis using LM (ngram or NN based)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nbest", type=str, help="Nbest hypothesis")
    parser.add_argument("lm", type=str, help="LM (ngram or NN based)")
    parser.add_argument("top1",
                        type=str,
                        help="The best hypothesis after rescoring")
    parser.add_argument("--lm-weight",
                        type=float,
                        default=0.1,
                        help="Language model weight")
    parser.add_argument("--lm-tag",
                        type=str,
                        default="best",
                        help="Tag name for NN based LM")
    parser.add_argument("--len-norm",
                        action=StrToBoolAction,
                        default=True,
                        help="If ture, using length normalized "
                        "for acoustic score")
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Path of the vocabulary dictionary")
    parser.add_argument("--spm",
                        type=str,
                        default="",
                        help="Path of the sentencepiece model "
                        "(for subword tokenizer)")
    parser.add_argument("--space",
                        type=str,
                        default="",
                        help="Space symbol to inserted between the "
                        "model units (if needed)")
    args = parser.parse_args()
    run(args)
