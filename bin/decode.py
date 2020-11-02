#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import codecs
import argparse

import torch as th

from aps.eval import Computer
from aps.opts import StrToBoolAction
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
        with codecs.open(args.dict, "r", encoding="utf-8") as f:
            vocab = {}
            for pair in f:
                unit, idx = pair.split()
                vocab[int(idx)] = unit
    else:
        vocab = None
    decoder = FasterDecoder(args.checkpoint, device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=16000,
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
                                  lm_weight=args.lm_weight,
                                  normalized=args.normalized,
                                  vectorized=args.vectorized)
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
        "Command to do End-To-End decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("best",
                        type=str,
                        help="Wspecifier for decoded results (1-best)")
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index for wav.scp")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the E2E model")
    parser.add_argument("--beam-size",
                        type=int,
                        default=8,
                        help="Beam size used during decoding")
    parser.add_argument("--lm",
                        type=str,
                        default="",
                        help="Checkpoint of the nerual network LM "
                        "used in shallow fusion")
    parser.add_argument("--lm-weight",
                        type=float,
                        default=0.1,
                        help="LM score weight used in shallow fusion")
    parser.add_argument("--dict",
                        type=str,
                        default="",
                        help="Dictionary file (not needed)")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--max-len",
                        type=int,
                        default=100,
                        help="Maximum steps to do during decoding stage")
    parser.add_argument("--space",
                        type=str,
                        default="",
                        help="space flag for language like EN "
                        "to merge characters to words")
    parser.add_argument("--nbest",
                        type=int,
                        default=1,
                        help="N-best decoded utterances to output")
    parser.add_argument("--dump-nbest",
                        type=str,
                        default="",
                        help="If not empty, dump n-best hypothesis")
    parser.add_argument("--function",
                        type=str,
                        choices=["beam_search", "greedy_search"],
                        default="beam_search",
                        help="Name of the decoding function")
    parser.add_argument("--vectorized",
                        action=StrToBoolAction,
                        default="true",
                        help="If ture, using vectorized algothrim")
    parser.add_argument("--normalized",
                        action=StrToBoolAction,
                        default="false",
                        help="If ture, using length normalized "
                        "when sort nbest hypos")
    args = parser.parse_args()
    run(args)
