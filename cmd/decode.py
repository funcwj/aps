#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import numpy as np
import torch as th

from pathlib import Path
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.opts import DecodingParser
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

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
    "min_len_ratio", "len_norm", "lm_weight", "temperature", "len_penalty",
    "cov_penalty", "eos_threshold", "cov_threshold", "allow_partial"
]


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
                                            task="asr",
                                            cpt_tag=cpt_tag,
                                            device_id=device_id)
        if not hasattr(self.nnet, function):
            raise RuntimeError(
                f"AM doesn't have the decoding function: {function}")
        self.decode = getattr(self.nnet, function)
        self.function = function
        logger.info(f"Load checkpoint from {cpt_dir}, epoch: " +
                    f"{self.epoch}, tag: {cpt_tag}")
        logger.info(f"Using decoding function: {function}")

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
            logger.info(f"Load RNN LM from {args.lm}: epoch {lm.epoch}, " +
                        f"weight = {args.lm_weight}")
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
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest_hypos = decoder.run(src, **dec_args)
        nbest = [f"{key}\n"]
        for idx, hyp in enumerate(nbest_hypos):
            # remove SOS/EOS
            token = hyp["trans"][1:-1]
            trans = processor.run(token)
            score = hyp["score"]
            nbest.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
            if idx == 0:
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
    logger.info(
        f"Decode {len(src_reader)} utterance done, time cost = {cost:.2f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do end-to-end decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DecodingParser.parser])
    parser.add_argument("--function",
                        type=str,
                        choices=["beam_search", "greedy_search"],
                        default="beam_search",
                        help="Name of the decoding function")
    args = parser.parse_args()
    run(args)
