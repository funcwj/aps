#!/usr/bin/env python
# wujian@2019

import argparse


def run(args):
    with open(args.dict, "rb") as f:
        pair = [line.decode("utf-8").split() for line in f]
        token2idx = {l: r for l, r in pair}
    with open(args.token, "rb") as f:
        for raw_line in f:
            tokens = raw_line.decode("utf-8").split()
            idx = []
            for tok in tokens[1:]:
                if tok not in token2idx:
                    tok = token2idx["<unk>"]
                else:
                    tok = token2idx[tok]
                idx.append(tok)
            print(tokens[0] + " " + " ".join(idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to map token to token-id",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("token", type=str, help="Token location")
    parser.add_argument("dict", type=str, help="Dictionary location")
    args = parser.parse_args()
    run(args)