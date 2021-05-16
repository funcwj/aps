#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import subprocess

from kaldi_python_io import Reader as BaseReader


def run(args):
    scp_out = open(args.scp, "w") if args.scp else None
    with open(args.wav_ark, "wb") as wav_ark:
        reader = BaseReader(args.wav_scp, num_tokens=2, restrict=True)
        done = 0
        for key, value in reader:
            wav_ark.write(str.encode(key + " "))
            offset = wav_ark.tell()
            succ = True
            if value[-1] == "|":
                p = subprocess.Popen(value[:-1],
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                [stdout, _] = p.communicate()
                if p.returncode != 0:
                    succ = False
                else:
                    wav_ark.write(stdout)
            else:
                try:
                    with open(value, "rb") as wav:
                        wav_ark.write(wav.read())
                except:
                    succ = False
            if not succ:
                continue
            if scp_out:
                scp_out.write(f"{key}\t{args.wav_ark}:{offset}\n")
            done += 1
            if done % 200 == 0:
                print(f"Processed {done}/{len(reader)} utterances...",
                      flush=True)
        print(f"Archive {done}/{len(reader)} utterances to {args.wav_ark}")
    if scp_out:
        scp_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do convert audio to Kaldi's .ark format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input audio script")
    parser.add_argument("wav_ark", type=str, help="Output audio archive")
    parser.add_argument("--scp",
                        type=str,
                        default="",
                        help="If assigned, generate corresponding .scp file")
    args = parser.parse_args()
    run(args)
