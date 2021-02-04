#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse
import warnings

import numpy as np
import torch as th

from pathlib import Path
from aps.opts import DecodingParser
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)

beam_search_params = [
    "beam_size", "nbest", "max_len", "min_len", "max_len_ratio",
    "min_len_ratio", "len_norm", "lm_weight", "temperature", "len_penalty",
    "cov_penalty", "eos_threshold", "cov_threshold", "allow_partial"
]


class BatchDecoder(NnetEvaluator):
    """
    Decoder wrapper
    """

    def __init__(self,
                 cpt_dir: str,
                 device_id: int = -1,
                 cpt_tag: str = "best") -> None:
        super(BatchDecoder, self).__init__(cpt_dir,
                                           task="asr",
                                           device_id=device_id,
                                           cpt_tag=cpt_tag)
        logger.info(f"Load checkpoint from {cpt_dir}, epoch: " +
                    f"{self.epoch}, tag: {cpt_tag}")

    def run(self, inps, **kwargs):
        return self.nnet.beam_search_batch(
            [th.from_numpy(t).to(self.device) for t in inps], **kwargs)


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    if args.batch_size == 1:
        warnings.warn("can use decode.py instead as batch_size == 1")
    decoder = BatchDecoder(args.am,
                           device_id=args.device_id,
                           cpt_tag=args.am_tag)
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
        nbest = min(args.beam_size, args.nbest)
        topn.write(f"{nbest}\n")
    ali_dir = args.dump_align
    if ali_dir:
        Path(ali_dir).mkdir(exist_ok=True, parents=True)
        logger.info(f"Dump alignments to dir: {ali_dir}")
    done = 0
    timer = SimpleTimer()
    batches = []
    dec_args = dict(
        filter(lambda x: x[0] in beam_search_params,
               vars(args).items()))
    dec_args["lm"] = lm
    for key, src in src_reader:
        done += 1
        batches.append({
            "key": key,
            "inp": src,
            "len": src.shape[-1] if decoder.accept_raw else src.shape[0]
        })
        end = (done == len(src_reader) and len(batches))
        if len(batches) != args.batch_size and not end:
            continue
        # decode
        batches = sorted(batches, key=lambda b: b["len"], reverse=True)
        batch_nbest = decoder.run([bz["inp"] for bz in batches], **dec_args)
        keys = [bz["key"] for bz in batches]
        for key, nbest in zip(keys, batch_nbest):
            logger.info(f"Decoding utterance {key}...")
            nbest_hypos = [f"{key}\n"]
            for idx, hyp in enumerate(nbest):
                # remove SOS/EOS
                token = hyp["trans"][1:-1]
                trans = processor.run(token)
                score = hyp["score"]
                nbest_hypos.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
                if idx == 0:
                    top1.write(f"{key}\t{trans}\n")
                if ali_dir:
                    if hyp["align"] is None:
                        raise RuntimeError(
                            "Can not dump alignment out as it's None")
                    np.save(f"{ali_dir}/{key}-nbest{idx+1}",
                            hyp["align"].numpy())
            if topn:
                topn.write("".join(nbest_hypos))
        top1.flush()
        if topn:
            topn.flush()
        batches.clear()

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
        "Command to do end-to-end decoding using batch version beam search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DecodingParser.parser])
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="Number of utterances to process in one batch")
    args = parser.parse_args()
    run(args)
