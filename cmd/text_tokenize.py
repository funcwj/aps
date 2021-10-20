#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Previous implementation is in aps/utils/tokenizer.{py,pl}
"""
import argparse

from aps.tokenizer import WordTokenizer, SubwordTokenizer
from aps.utils import io_wrapper, get_logger

logger = get_logger(__name__)


def add_to_vocab(vocab, units):
    if vocab is None:
        return
    for unit in units:
        if unit not in vocab:
            vocab[unit] = len(vocab)


def run(args):
    src_std, src = io_wrapper(args.src_txt, "r")
    dst_std, dst = io_wrapper(args.dst_tok, "w")

    filter_units = args.filter_units.split(",")
    logger.info(f"Filter units: {filter_units}")

    vocab = None
    add_units = None

    if args.add_units:
        add_units = args.add_units.split(",")
    if args.dump_vocab:
        vocab = {}
        if add_units:
            logger.info(f"Add units: {add_units} to vocabulary")
            add_to_vocab(vocab, add_units)
        if args.space:
            logger.info(f"Add units: {args.space} to vocabulary")
            add_to_vocab(vocab, [args.space])

    if args.unit == "subword":
        if not args.spm:
            raise RuntimeError("Missing --spm when choose subword unit")
        tokenizer = SubwordTokenizer(args.spm, filter_words=filter_units)
    else:
        use_char = args.unit == "char"
        tokenizer = WordTokenizer(filter_units, char=use_char, space=args.space)
    for raw_line in src:
        line = raw_line.strip()
        raw_tokens = line.split()
        if args.text_format == "kaldi":
            sets = raw_tokens[1:]
            dst.write(f"{raw_tokens[0]}\t")
        else:
            sets = raw_tokens
        tokens = tokenizer.run(sets)
        add_to_vocab(vocab, tokens)
        dst.write(" ".join(tokens) + "\n")
    if vocab:
        _, dump_vocab = io_wrapper(args.dump_vocab, "w")
        for unit, idx in vocab.items():
            dump_vocab.write(f"{unit} {idx}\n")
        logger.info(
            f"Dump vocabulary to {args.dump_vocab} with {len(vocab)} units")
        dump_vocab.close()
    if not src_std:
        src.close()
    if not dst_std:
        dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize the text to the modeling units, e.g., "
        "character, phoneme, word, subword, ...")
    parser.add_argument("src_txt",
                        type=str,
                        help="Source text file (Kaldi format or not)")
    parser.add_argument("dst_tok",
                        type=str,
                        help="Output text file (Kaldi format or not)")
    parser.add_argument("--text-format",
                        type=str,
                        default="kaldi",
                        choices=["kaldi", "raw"],
                        help="Format of the text file. "
                        "The kaldi format begins with the utterance ID")
    parser.add_argument("--spm",
                        type=str,
                        default="",
                        help="Path of the sentencepiece's model "
                        "if we choose subword unit")
    parser.add_argument("--filter-units",
                        type=str,
                        default="",
                        help="Filter the units if needed, "
                        "each unit is separated via \',\'")
    parser.add_argument("--unit",
                        type=str,
                        default="char",
                        choices=["char", "word", "subword"],
                        help="Type of the modeling unit")
    parser.add_argument("--space",
                        type=str,
                        default="",
                        help="If not none, insert space "
                        "symbol between each units")
    parser.add_argument("--add-units",
                        type=str,
                        default="",
                        help="Add units to vocabulary set, "
                        "e.g., <sos>, <eos>, <unk>")
    parser.add_argument("--dump-vocab",
                        type=str,
                        default="",
                        help="If not none, dump out the vocabulary set")
    args = parser.parse_args()
    run(args)
