#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

import torch as th

from aps.eval import Computer
from aps.opts import DecodingParser
from aps.conf import load_dict
from aps.utils import get_logger, io_wrapper
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)
"""
Nbest format:
Number: n
key1
score-1 hyp-1
...
score-n hyp-n
...
keyM
score-1 hyp-1
...
score-n hyp-n
"""


class FasterDecoder(Computer):
    """
    Decoder wrapper
    """

    def __init__(self, cpt_dir, function="beam_search", device_id=-1):
        super(FasterDecoder, self).__init__(cpt_dir, device_id=device_id)
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
    # build dictionary
    if args.dict:
        vocab = load_dict(args.dict, reverse=True)
    else:
        vocab = None
    decoder = FasterDecoder(args.checkpoint,
                            function=args.function,
                            device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=args.sr,
                                 channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    if args.lm:
        lm = Computer(args.lm, device_id=args.device_id)
        logger.info(f"Load lm from {args.lm}: epoch {lm.epoch}, " +
                    f"weight = {args.lm_weight}")
        lm = lm.nnet
    else:
        lm = None

    stdout_top1, top1 = io_wrapper(args.best, "w")
    topn = None
    if args.dump_nbest:
        stdout_topn, topn = io_wrapper(args.dump_nbest, "w")
        topn.write(f"{args.nbest}\n")

    N = 0
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest_hypos = decoder.run(src,
                                  lm=lm,
                                  beam=args.beam_size,
                                  nbest=args.nbest,
                                  max_len=args.max_len,
                                  penalty=args.penalty,
                                  lm_weight=args.lm_weight,
                                  normalized=args.normalized,
                                  temperature=args.temperature)
        nbest = [f"{key}\n"]
        for idx, hyp in enumerate(nbest_hypos):
            score = hyp["score"]
            # remove SOS/EOS
            if vocab:
                trans = [vocab[idx] for idx in hyp["trans"][1:-1]]
            else:
                trans = [str(idx) for idx in hyp["trans"][1:-1]]
            if vocab and args.space:
                trans = "".join(trans).replace(args.space, " ")
            else:
                trans = " ".join(trans)
            nbest.append(f"{score:.3f}\t{trans}\n")
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
    logger.info(f"Decode {len(src_reader)} utterance done")


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
