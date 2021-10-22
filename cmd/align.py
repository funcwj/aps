#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import torch as th
import numpy as np

from aps.opts import AlignmentParser
from aps.eval import NnetEvaluator, TextPreProcessor
from aps.conf import load_dict
from aps.utils import get_logger, SimpleTimer
from aps.io import AudioReader, SegmentAudioReader, io_wrapper

from kaldi_python_io import ScriptReader, Reader
from typing import Dict, Optional

logger = get_logger(__name__)
"""
Alignment format:
utt-1 <alignment string>
utt-2 <alignment string>
...
utt-N <alignment string>
"""


class CtcAligner(NnetEvaluator):
    """
    Wrapper for CTC force alignment
    """

    def __init__(self,
                 cpt_dir: str,
                 cpt_tag: str = "best",
                 device_id: int = -1) -> None:
        super(CtcAligner, self).__init__(cpt_dir,
                                         cpt_tag=cpt_tag,
                                         device_id=device_id)
        logger.info(f"Load the checkpoint from {cpt_dir}, epoch: " +
                    f"{self.epoch}, tag: {cpt_tag}")

    def run(self, inp: np.ndarray, seq: np.ndarray) -> Dict:
        inp = th.from_numpy(inp).to(self.device)
        seq = th.tensor(seq, dtype=th.int64).to(self.device)
        return self.nnet.ctc_align(inp, seq)


def gen_word_boundary(key: str,
                      dur: float,
                      ali_str: str,
                      vocab: Optional[Dict] = None):
    # e.g., * * * * * * * * * * 5464 * * * * * *
    ali_seq = ali_str.split(" ")
    dur_per_frame = dur / len(ali_seq)
    # non-blank info
    ali_pos = [((f + 1) * dur_per_frame, tok)
               for f, tok in enumerate(ali_seq)
               if tok != "*"]
    boundary = [f"{key} {len(ali_seq)} {len(ali_pos)} {dur:.3f}"]
    beg = 0
    for pos in ali_pos:
        end, tok = pos
        if vocab:
            tok = vocab[int(tok)]
        boundary.append(f"{key} {tok} {beg:.3f} {end:.3f}")
        beg = end
    return boundary


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)

    aligner = CtcAligner(args.am, cpt_tag=args.am_tag, device_id=args.device_id)
    if aligner.accept_raw:
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
        if args.word_boundary:
            raise RuntimeError(
                "Now can't generate word boundary when using Kaldi's feature")

    txt_reader = Reader(args.text, num_tokens=-1, restrict=False)
    processor = TextPreProcessor(args.dict, space=args.space, spm=args.spm)

    ali_stdout, ali_fd = io_wrapper(args.alignment, "w")

    wdb_stdout, wdb_fd = False, None
    vocab_dict = None
    if args.word_boundary:
        if args.dict:
            vocab_dict = load_dict(args.dict, reverse=True)
        wdb_stdout, wdb_fd = io_wrapper(args.word_boundary, "w")
    done = 0
    tot_utts = len(src_reader)
    timer = SimpleTimer()
    for key, str_seq in txt_reader:
        done += 1
        logger.info(
            f"Generate alignment for utterance {key} ({done}/{tot_utts}) ...")
        int_seq = processor.run(str_seq)
        wav_or_feats = src_reader[key]
        ali = aligner.run(wav_or_feats, int_seq)
        header = f"{ali['score']:.3f}, {len(ali['align_seq'])}"
        ali_fd.write(f"{key} {ali['align_str']}\n")
        logger.info(f"{key} ({header}) {ali['align_str']}")
        if wdb_fd:
            dur = wav_or_feats.shape[-1] * 1.0 / args.sr
            wdb = gen_word_boundary(key,
                                    dur,
                                    ali["align_str"],
                                    vocab=vocab_dict)
            wdb_fd.write("\n".join(wdb) + "\n")
    if not ali_stdout:
        ali_fd.close()
    if wdb_fd and not wdb_stdout:
        wdb_fd.close()
    cost = timer.elapsed()
    logger.info(f"Generate alignments for {tot_utts} utterance done, " +
                f"time cost = {cost:.2f}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do force alignment (currently uses CTC branch in E2E AM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[AlignmentParser.parser])
    args = parser.parse_args()
    run(args)
