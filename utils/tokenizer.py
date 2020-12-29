#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Python version of tokenizer.pl
"""

import sys
import codecs
import argparse


def io_wrapper(io_str, mode):
    """
    Wrapper for IO stream
    """
    if io_str != "-":
        std = False
        stream = codecs.open(io_str, mode, encoding="utf-8")
    else:
        std = True
        if mode not in ["r", "w"]:
            raise RuntimeError(f"Unknown IO mode: {mode}")
        if mode == "w":
            stream = codecs.getwriter("utf-8")(sys.stdout.buffer)
        else:
            stream = codecs.getreader("utf-8")(sys.stdin.buffer)
    return std, stream


def run(args):
    src_std, src = io_wrapper(args.src_txt, "r")
    dst_std, dst = io_wrapper(args.dst_tok, "w")

    def add_to_vocab(vocab, units):
        for unit in units:
            if unit not in vocab:
                vocab[unit] = len(vocab)

    sp_mdl = None
    vocab = None
    add_units = None
    if args.unit == "subword":
        if not args.spm:
            raise RuntimeError("Missing --spm when choose subword unit")
        import sentencepiece as sp
        sp_mdl = sp.SentencePieceProcessor(model_file=args.spm)
    else:
        if args.add_units:
            add_units = args.add_units.split(",")
        if args.dump_vocab:
            vocab = {}
            if add_units:
                print(f"Add units: {add_units} to vocabulary")
                add_to_vocab(vocab, add_units)
        if args.space:
            add_to_vocab(vocab, [args.space])
    filter_units = args.filter_units.split(",")
    for raw_line in src:
        line = raw_line.strip()
        raw_tokens = line.split()
        if args.text_format == "kaldi":
            sets = raw_tokens[1:]
            dst.write(f"{raw_tokens[0]}\t")
        else:
            sets = raw_tokens
        kept_tokens = []
        for tok in sets:
            # remove tokens
            if tok in filter_units:
                continue
            # word => char
            if args.unit == "char":
                toks = [t for t in tok]
            else:
                toks = [tok]
            kept_tokens += toks
            if vocab is not None:
                add_to_vocab(vocab, toks)
        if args.unit == "subword":
            kept_tokens = sp_mdl.encode(" ".join(kept_tokens), out_type=str)
        if args.space:
            dst.write(f" {args.space} ".join(kept_tokens) + "\n")
        else:
            dst.write(" ".join(kept_tokens) + "\n")
    if vocab:
        _, dump_vocab = io_wrapper(args.dump_vocab, "w")
        for unit, idx in vocab.items():
            dump_vocab.write(f"{unit} {idx}\n")
        print(f"Dump vocabulary to {args.dump_vocab} with {len(vocab)} units")
        dump_vocab.close()
    if not src_std:
        src.close()
    if not dst_std:
        dst.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize the text to modeling units, e.g., "
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
