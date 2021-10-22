#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import math
import argparse

from aps.opts import StrToBoolAction
from aps.io import TextReader
from aps.metric.reporter import WerReporter
from aps.metric.asr import permute_wer


class TransReader(object):
    """
    Class to handle single/multi-speaker transcriptions
    """

    def __init__(self, descriptor, cer=False):
        self.readers = [
            TextReader(td, char=cer) for td in descriptor.split(",")
        ]

    def __len__(self):
        return len(self.readers)

    def __getitem__(self, key):
        if not self._check(key):
            raise RuntimeError(f"Missing {key} in one of the text files")
        return [reader[key] for reader in self.readers]

    def _check(self, key):
        status = [key in reader for reader in self.readers]
        return sum(status) == len(self.readers)

    def __iter__(self):
        ref = self.readers[0]
        for key in ref.index_keys:
            if self._check(key):
                yield key, self[key]


def run(args):
    hyp_reader = TransReader(args.hyp, cer=args.cer)
    ref_reader = TransReader(args.ref, cer=args.cer)
    if len(hyp_reader) != len(ref_reader):
        raise RuntimeError("Looks number of speakers do not match in hyp & ref")
    each_utt = open(args.per_utt, "w") if args.per_utt else None

    reporter = WerReporter(args.utt2class,
                           name="CER" if args.cer else "WER",
                           unit="%")
    for key, hyp in hyp_reader:
        ref = ref_reader[key]
        if args.reduce == "sum" or len(hyp_reader) == 1:
            err = permute_wer(hyp, ref)
            ref_len = sum([len(r) for r in ref])
        else:
            err = [math.inf, 0, 0]
            ref_len = None, None
            for h, r in zip(hyp, ref):
                cur_err = permute_wer([h], [r])
                if sum(cur_err) < sum(err):
                    err = cur_err
                    ref_len = len(r)
        if each_utt:
            if ref_len != 0:
                each_utt.write(f"{key}\t{sum(err) / ref_len:.3f}\n")
            else:
                each_utt.write(f"{key}\tINF\n")
        reporter.add(key, err, ref_len)
    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute wer (edit/levenshtein distance), "
        "accepting text following Kaldi's format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("hyp",
                        type=str,
                        help="Hypothesis transcripts "
                        "(multi-speakers need split by ',')")
    parser.add_argument("ref",
                        type=str,
                        help="References transcripts "
                        "(multi-speakers need split by ',')")
    parser.add_argument("--per-utt",
                        type=str,
                        default="",
                        help="If assigned, compute wer for each utterance")
    parser.add_argument("--utt2class",
                        type=str,
                        default="",
                        help="If assigned, report results "
                        "per-class (gender or degree)")
    parser.add_argument("--cer",
                        action=StrToBoolAction,
                        default=False,
                        help="Compute CER instead of WER")
    parser.add_argument("--reduce",
                        type=str,
                        choices=["sum", "min"],
                        default="sum",
                        help="Reduction options for multi-speaker cases")
    args = parser.parse_args()
    run(args)
