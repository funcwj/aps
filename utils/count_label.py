#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Count number of the units
"""
import codecs
import warnings
import argparse


def run(args):
    id2unit = {}
    unit2id = {}
    with codecs.open(args.dict, "r", encoding="utf-8") as dict_fd:
        for raw_line in dict_fd:
            unit, idx = raw_line.strip().split()
            unit2id[unit] = int(idx)
            id2unit[int(idx)] = unit

    counts = [0] * len(id2unit)
    num_unk = 0
    num_tot = 0
    with codecs.open(args.text, "r", encoding="utf-8") as text_fd:
        for raw_line in text_fd:
            toks = raw_line.strip().split()
            for tok in toks[1:]:
                num_tot += len(toks[1:])
                if tok in unit2id:
                    counts[unit2id[tok]] += 1
                else:
                    num_unk += 1
    if num_unk:
        ratio = num_unk * 100.0 / num_tot
        warnings.warn(f"Got {num_unk} {args.unk_tok} ({ratio:.4f}%) tokens")
    for i, c in enumerate(counts):
        print(f"{id2unit[i]} {c}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to output the number of the units in the "
        "given transcription file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("dict", type=str, help="Dictionary used for training")
    parser.add_argument("text", type=str, help="Transcription file")
    parser.add_argument("--unk-tok",
                        type=str,
                        default="<unk>",
                        help="Token name for unknown tokens")
    args = parser.parse_args()
    run(args)
