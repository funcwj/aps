#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
import sys
import argparse
import warnings
import subprocess

import soundfile as sf


def ext_open(fd, mode):
    if mode not in ["r", "w"]:
        raise ValueError(f"Unsupported mode: {mode}")
    if fd == "-":
        return sys.stdout if mode == "w" else sys.stdin
    else:
        return open(fd, mode)


def run(args):
    prog_interval = 100
    done, total = 0, 0
    utt2dur = ext_open(args.utt2dur, "w")
    wav_scp = ext_open(args.wav_scp, "r")
    for raw_line in wav_scp:
        total += 1
        line = raw_line.strip()
        toks = line.split()
        succ = True
        if line[-1] == "|":
            key, cmd = toks[0], " ".join(toks[1:-1])
            p = subprocess.Popen(cmd,
                                 shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            [stdout, stderr] = p.communicate()
            if p.returncode != 0:
                stderr = bytes.decode(stderr)
                raise RuntimeError(
                    f"Running command: \"{cmd}\" failed: {stderr}")
            wav_io = io.BytesIO(stdout)
            data, sr = sf.read(wav_io, dtype="int16")
            dur = data.shape[0]
            if args.output == "time":
                dur = float(dur) / sr
        else:
            if len(toks) != 2:
                warnings.warn(f"Line format error: {line}")
                continue
            key, path = toks
            try:
                info = sf.info(path)
                dur = info.duration if args.output == "time" else info.frames
            except:
                succ = False
                print(f"Failed to work out duration of utterance {key} ...")
        if not succ:
            continue
        done += 1
        if args.output == "time":
            utt2dur.write(f"{key}\t{dur:.4f}\n")
        else:
            utt2dur.write(f"{key}\t{dur:d}\n")
        if done % prog_interval == 0:
            print(f"Processed {done}/{total} utterances...", flush=True)
    if args.utt2dur != "-":
        utt2dur.close()
    if args.wav_scp != "-":
        wav_scp.close()
    print(f"Processed {done} utterances done, total {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to generate duration of the wave. "
        "We avoid to read whole utterance as it may slow down the speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input wave script")
    parser.add_argument("utt2dur", type=str, help="Output utt2dur file")
    parser.add_argument("--output",
                        type=str,
                        choices=["time", "sample"],
                        default="sample",
                        help="Output type of the script")
    args = parser.parse_args()
    run(args)
