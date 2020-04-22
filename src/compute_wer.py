#!/usr/bin/env python

# wujian@2019

import argparse

import editdistance as ed

from collections import defaultdict
from itertools import permutations
from libs.utils import StrToBoolAction
from kaldi_python_io import Reader as BaseReader


def permute_ed(hlist, rlist):
    """
    Compute edit distance between N pairs
    args:
        hlist: list[vector], hypothesis
        rlist: list[vector], reference 
    """
    def distance(hlist, rlist):
        return sum([ed.eval(h, r) for h, r in zip(hlist, rlist)])

    N = len(hlist)
    if N != len(rlist):
        raise RuntimeError("size do not match between hlist "
                           "and rlist: {:d} vs {:d}".format(N, len(rlist)))
    wers = []
    for order in permutations(range(N)):
        wers.append(distance(hlist, [rlist[n] for n in order]))
    return min(wers)


class TransReader(object):
    """
    Class to handle single/multi-speaker transcriptions
    """
    def __init__(self, text_descriptor):
        self.readers = [
            BaseReader(t, num_tokens=-1, restrict=False)
            for t in text_descriptor.split(",")
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


class Report(object):
    def __init__(self, spk2class=None):
        self.s2c = BaseReader(spk2class) if spk2class else None
        self.err = defaultdict(float)
        self.tot = defaultdict(float)
        self.cnt = 0

    def add(self, key, err, tot):
        cls_str = "NG"
        if self.s2c:
            cls_str = self.s2c[key]
        self.err[cls_str] += err
        self.tot[cls_str] += tot
        self.cnt += 1

    def report(self):
        print("WER(%) Report: ")
        sum_err = sum([self.err[cls_str] for cls_str in self.err])
        sum_len = sum([self.tot[cls_str] for cls_str in self.tot])
        print(f"Total WER: {sum_err * 100 / sum_len:.2f}%, " +
              f"{self.cnt} utterances")
        if len(self.err) != 1:
            for cls_str in self.err:
                cls_err = self.err[cls_str]
                cls_tot = self.tot[cls_str]
                print(f"  {cls_str}: {cls_err * 100 / cls_tot:.2f}%")


def run(args):
    hyp_reader = TransReader(args.hyp)
    ref_reader = TransReader(args.ref)
    if len(hyp_reader) != len(ref_reader):
        raise RuntimeError(
            "Looks number of speakers do not match in hyp & ref")
    each_utt = open(args.per_utt, "w") if args.per_utt else None

    reporter = Report(args.utt2class)
    for key, hyp in hyp_reader:
        ref = ref_reader[key]
        err = permute_ed(hyp, ref)
        ref_len = sum([len(r) for r in ref])
        if each_utt:
            if ref_len != 0:
                each_utt.write("{}\t{:.3f}\n".format(key, err / ref_len))
            else:
                each_utt.write("{}\tINF\n".format(key))
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
    args = parser.parse_args()
    run(args)