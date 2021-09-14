#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
import os
import sys
import argparse
import warnings
import subprocess

import soundfile as sf
import multiprocessing as mp

prog_interval = 100


def ext_open(fd, mode):
    if mode not in ["r", "w"]:
        raise ValueError(f"Unsupported mode: {mode}")
    if fd == "-":
        return sys.stdout if mode == "w" else sys.stdin
    else:
        return open(fd, mode)


def worker(jobid, num_jobs, wav_scp, utt2dur, output):
    """
    Define a single processing worker
    """
    wav_scp_fp = ext_open(wav_scp, "r")
    utt2dur_fp = ext_open(utt2dur, "w")
    done, total = 0, 0
    for raw_line in wav_scp_fp:
        total += 1
        if (total - 1) % num_jobs != jobid:
            continue
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
                raise RuntimeError(f"Worker {jobid}: running command: " +
                                   f"\"{cmd}\" failed: {stderr}")
            wav_io = io.BytesIO(stdout)
            data, sr = sf.read(wav_io, dtype="int16")
            dur = data.shape[0]
            if args.output == "time":
                dur = float(dur) / sr
        else:
            if len(toks) != 2:
                warnings.warn(f"Worker {jobid}: line format error: {line}")
                continue
            key, path = toks
            try:
                info = sf.info(path)
                dur = info.duration if output == "time" else info.frames
            except RuntimeError:
                succ = False
                print(f"Worker {jobid}: failed to work out " +
                      f"duration of utterance {key} ...")
        if not succ:
            continue
        done += 1
        if output == "time":
            utt2dur_fp.write(f"{key}\t{dur:.4f}\n")
        else:
            utt2dur_fp.write(f"{key}\t{dur:d}\n")
        if done % prog_interval == 0:
            print(f"Worker {jobid}: processed {done} utterances...", flush=True)
    if utt2dur != "-":
        utt2dur_fp.close()
    if wav_scp != "-":
        wav_scp_fp.close()
    print(f"Worker {jobid}: processed {done}/{total} utterances done",
          flush=True)


def run(args):
    if args.wav_scp == "-" or args.utt2dur == "-" or args.num_jobs <= 1:
        worker(0, 1, args.wav_scp, args.utt2dur, args.output)
    else:
        wav2dur_list = []
        pool = mp.Pool(args.num_jobs)
        for j in range(args.num_jobs):
            wav2dur_list.append(f"{args.utt2dur}.{j}")
            packed_args = (j, args.num_jobs, args.wav_scp, wav2dur_list[-1],
                           args.output)
            pool.apply_async(worker, args=packed_args)
        pool.close()
        pool.join()
        wav2dur_list = " ".join(wav2dur_list)
        merge_cmd = f"cat {wav2dur_list} | sort -k1 > {args.utt2dur} && rm {wav2dur_list}"
        os.system(merge_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to generate duration of the wave. "
        "We avoid to read whole utterance as it may slow down the speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input wave script")
    parser.add_argument("utt2dur", type=str, help="Output utt2dur file")
    parser.add_argument("--num-jobs",
                        type=int,
                        default=1,
                        help="Number of the parallel jobs to run")
    parser.add_argument("--output",
                        type=str,
                        choices=["time", "sample"],
                        default="sample",
                        help="Output type of the script")
    args = parser.parse_args()
    run(args)
