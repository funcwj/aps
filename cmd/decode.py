#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import torch as th

from pathlib import Path
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.opts import DecodingParser
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)
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
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")
        logger.info(f"Using decoding function: {function}")

    def run(self, src, **kwargs):
        src = th.from_numpy(src).to(self.device)
        if self.function == "greedy_search":
            return self.decode(src)
        else:
            return self.decode(src, **kwargs)


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)

    decoder = FasterDecoder(args.checkpoint,
                            cpt_tag=args.tag,
                            function=args.function,
                            device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=args.sr,
                                 norm=args.wav_norm,
                                 channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    if args.lm:
        if Path(args.lm).is_file():
            from aps.asr.lm.ngram import NgramLM
            lm = NgramLM(args.lm, args.dict)
            logger.info(f"Load ngram from {args.lm}, weight = {args.lm_weight}")
        else:
            lm = NnetEvaluator(args.lm, device_id=args.device_id)
            logger.info(f"Load rnnlm from {args.lm}: epoch {lm.epoch}, " +
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
        nbest = min(args.beam_size, args.nbest)
        topn.write(f"{nbest}\n")

    N = 0
    timer = SimpleTimer()
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest_hypos = decoder.run(src,
                                  lm=lm,
                                  beam=args.beam_size,
                                  nbest=args.nbest,
                                  max_len=args.max_len,
                                  penalty=args.penalty,
                                  lm_weight=args.lm_weight,
                                  len_norm=args.len_norm,
                                  temperature=args.temperature)
        nbest = [f"{key}\n"]
        for idx, hyp in enumerate(nbest_hypos):
            # remove SOS/EOS
            token = hyp["trans"][1:-1]
            trans = processor.run(token)
            score = hyp["score"]
            nbest.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
            if idx == 0:
                top1.write(f"{key}\t{trans}\n")
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
        f"Decode {len(src_reader)} utterance done, time cost = {cost:.2f}s")


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
