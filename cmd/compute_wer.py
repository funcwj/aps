#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

from aps.opts import StrToBoolAction
from aps.metric.reporter import WerReporter
from aps.metric.asr import permute_wer

from kaldi_python_io import Reader as BaseReader


def to_chars(str_list):
    """
    Convert str list to char list
    """
    chars = []
    for sstr in str_list:
        chars += list(sstr)
    return chars


class TransReader(object):
    """
    Class to handle single/multi-speaker transcriptions
    """

    def __init__(self, text_descriptor, cer=False):
        self.readers = [
            BaseReader(t, num_tokens=-1, restrict=False)
            for t in text_descriptor.split(",")
        ]
        self.cer = cer

    def __len__(self):
        return len(self.readers)

    def __getitem__(self, key):
        if not self._check(key):
            raise RuntimeError(f"Missing {key} in one of the text files")
        trans = [reader[key] for reader in self.readers]
        if self.cer:
            trans = [to_chars(r) for r in trans]
        return trans

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
        err = permute_wer(hyp, ref)
        ref_len = sum([len(r) for r in ref])
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
    args = parser.parse_args()
    run(args)
