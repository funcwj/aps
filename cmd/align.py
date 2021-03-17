#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

import torch as th
import numpy as np

from aps.opts import AlignmentParser
from aps.eval import NnetEvaluator, TextPreProcessor
from aps.utils import get_logger, io_wrapper, SimpleTimer
from aps.loader import AudioReader

from kaldi_python_io import ScriptReader, Reader
from typing import Dict

logger = get_logger(__name__)
"""
Alignment format:
utt-1 score <align>
utt-2 score <align>
...
utt-N score <align>
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


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)

    aligner = CtcAligner(args.am, cpt_tag=args.am_tag, device_id=args.device_id)
    if aligner.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=args.sr,
                                 channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    txt_reader = Reader(args.text, num_tokens=-1, restrict=False)
    processor = TextPreProcessor(args.dict, space=args.space, spm=args.spm)

    stdout_ali, ali_fd = io_wrapper(args.alignment, "w")
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
        logger.info(f"{key} {header} {ali['align_str']}")
    if not stdout_ali:
        ali_fd.close()
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
