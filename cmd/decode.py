#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import numpy as np
import torch as th

from pathlib import Path

from aps.io import AudioReader, SegmentAudioReader, io_wrapper
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.opts import DecodingParser
from aps.conf import load_dict
from aps.const import UNK_TOKEN
from aps.utils import get_logger, SimpleTimer

from kaldi_python_io import ScriptReader
"""
Nbest format:
Number: n
key1
score-1 num-tok-in-hyp-1 hyp-1
...
score-n num-tok-in-hyp-n hyp-n
...
keyM
score-1 num-tok-in-hyp-1 hyp-1
...
score-n num-tok-in-hyp-n hyp-n
"""

logger = get_logger(__name__)

beam_search_params = [
    "beam_size", "nbest", "max_len", "min_len", "max_len_ratio",
    "min_len_ratio", "len_norm", "lm_weight", "ctc_weight", "temperature",
    "len_penalty", "cov_penalty", "eos_threshold", "cov_threshold",
    "allow_partial", "end_detect"
]

function_choices = ["beam_search", "greedy_search"]


class FasterDecoder(NnetEvaluator):
    """
    Decoder wrapper
    """

    def __init__(self,
                 cpt_dir: str,
                 cpt_tag: str = "best",
                 function: str = "beam_search",
                 device_id: int = -1) -> None:
        super(FasterDecoder, self).__init__(cpt_dir,
                                            cpt_tag=cpt_tag,
                                            device_id=device_id)
        if not hasattr(self.nnet, function):
            raise RuntimeError(
                f"AM doesn't have the decoding function: {function}")
        self.decode = getattr(self.nnet, function)
        self.function = function
        logger.info(f"Use decoding function: {function}")

    def run(self, src, **kwargs):
        src = th.from_numpy(src).to(self.device)
        if self.function == "greedy_search":
            return self.decode(src)
        else:
            return self.decode(src, **kwargs)


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)

    decoder = FasterDecoder(args.am,
                            cpt_tag=args.am_tag,
                            function=args.function,
                            device_id=args.device_id)
    if decoder.accept_raw:
        if args.segment:
            src_reader = SegmentAudioReader(args.feats_or_wav_scp,
                                            args.segment,
                                            sr=args.sr,
                                            channel=args.channel)
        else:
            src_reader = AudioReader(args.feats_or_wav_scp,
                                     sr=args.sr,
                                     channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    if args.lm:
        if Path(args.lm).is_file():
            from aps.asr.lm.ngram import NgramLM
            lm = NgramLM(args.lm, args.dict)
            logger.info(
                f"Load ngram LM from {args.lm}, weight = {args.lm_weight}")
        else:
            lm = NnetEvaluator(args.lm,
                               device_id=args.device_id,
                               cpt_tag=args.lm_tag)
            logger.info(f"Load NN LM from {args.lm}: weight = {args.lm_weight}")
            lm = lm.nnet
    else:
        lm = None

    processor = TextPostProcessor(args.dict,
                                  space=args.space,
                                  show_unk=args.show_unk,
                                  spm=args.spm)
    stdout_top1, top1 = io_wrapper(args.best, "w")
    topn = None
    if args.dump_nbest:
        stdout_topn, topn = io_wrapper(args.dump_nbest, "w")
        if args.function == "greedy_search":
            nbest = 1
        else:
            nbest = min(args.beam_size, args.nbest)
        topn.write(f"{nbest}\n")
    ali_dir = args.dump_align
    if ali_dir:
        Path(ali_dir).mkdir(exist_ok=True, parents=True)
        logger.info(f"Dump alignments to dir: {ali_dir}")
    N = 0
    timer = SimpleTimer()
    dec_args = dict(
        filter(lambda x: x[0] in beam_search_params,
               vars(args).items()))
    dec_args["lm"] = lm
    unk_idx = -1
    if args.dict and args.disable_unk:
        vocab_dict = load_dict(args.dict)
        if UNK_TOKEN in vocab_dict:
            unk_idx = vocab_dict[UNK_TOKEN]
            logger.info(f"Use unknown token {UNK_TOKEN} index: {unk_idx}")
    dec_args["unk"] = unk_idx
    done = 0
    tot_utts = len(src_reader)
    for key, src in src_reader:
        done += 1
        logger.info(f"Decoding utterance {key} ({done}/{tot_utts}) ...")
        nbest_hypos = decoder.run(src, **dec_args)
        nbest = [f"{key}\n"]
        for idx, hyp in enumerate(nbest_hypos):
            # remove SOS/EOS
            token = hyp["trans"][1:-1]
            trans = processor.run(token)
            score = hyp["score"]
            nbest.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
            if idx == 0:
                logger.info(f"{key} ({score:.3f}, {len(token):d}) {trans}")
                top1.write(f"{key}\t{trans}\n")
            if ali_dir:
                if hyp["align"] is None:
                    raise RuntimeError(
                        "Can not dump alignment out as it's None")
                np.save(f"{ali_dir}/{key}-nbest{idx+1}", hyp["align"].numpy())
        if topn:
            topn.write("".join(nbest))
        if not (N + 1) % 10:
            top1.flush()
            if topn:
                topn.flush()
        N += 1
    if not stdout_top1:
        top1.close()
    if topn and not stdout_topn:
        topn.close()
    cost = timer.elapsed()
    logger.info(f"Decode {tot_utts} utterance done, time cost = {cost:.2f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do end-to-end decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DecodingParser.parser])
    parser.add_argument("--function",
                        type=str,
                        choices=function_choices,
                        default="beam_search",
                        help="Name of the decoding function")
    args = parser.parse_args()
    run(args)
