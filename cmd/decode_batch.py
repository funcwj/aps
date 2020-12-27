#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse
import warnings

import torch as th

from aps.opts import DecodingParser
from aps.eval import NnetEvaluator, TextPostProcessor
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class BatchDecoder(NnetEvaluator):
    """
    Decoder wrapper
    """

    def __init__(self, cpt_dir, device_id=-1):
        super(BatchDecoder, self).__init__(cpt_dir, device_id=device_id)
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def run(self, inps, **kwargs):
        return self.nnet.beam_search_batch(
            [th.from_numpy(t).to(self.device) for t in inps], **kwargs)


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    if args.batch_size == 1:
        warnings.warn("can use decode.py instead as batch_size == 1")
    decoder = BatchDecoder(args.checkpoint, device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=args.sr,
                                 channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    if args.lm:
        lm = NnetEvaluator(args.lm, device_id=args.device_id)
        logger.info(f"Load lm from {args.lm}: epoch {lm.epoch}, " +
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

    done = 0
    timer = SimpleTimer()
    batches = []
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
        batch_nbest = decoder.run([bz["inp"] for bz in batches],
                                  lm=lm,
                                  beam=args.beam_size,
                                  nbest=args.nbest,
                                  max_len=args.max_len,
                                  penalty=args.penalty,
                                  lm_weight=args.lm_weight,
                                  normalized=args.normalized,
                                  temperature=args.temperature)
        keys = [bz["key"] for bz in batches]
        for key, nbest in zip(keys, batch_nbest):
            logger.info(f"Decoding utterance {key}...")
            nbest_hypos = [f"{key}\n"]
            for idx, hyp in enumerate(nbest):
                # remove SOS/EOS
                token = hyp["trans"][1:-1]
                trans = processor.run(token)
                score = hyp["score"]
                nbest.append(f"{score:.3f}\t{len(token):d}\t{trans}\n")
                if idx == 0:
                    top1.write(f"{key}\t{trans}\n")
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
        f"Decode {len(src_reader)} utterance done, time cost = {cost:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do end-to-end decoding using batch version beam search "
        "(WER may different with non-batch version when #batch_size != 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DecodingParser.parser])
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="Number of utterances to process in one batch")
    args = parser.parse_args()
    run(args)
