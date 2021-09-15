#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
import subprocess
import multiprocessing as mp

from kaldi_python_io import Reader as BaseReader


def worker(jobid, num_jobs, wav_scp, scp_out, ark_out):
    scp_out = open(scp_out, "w")
    with open(ark_out, "wb") as wav_ark:
        reader = BaseReader(wav_scp, num_tokens=2, restrict=True)
        done, n = 0, 0
        for key, value in reader:
            n += 1
            if (n - 1) % num_jobs != jobid:
                continue
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
                except FileNotFoundError:
                    succ = False
                    print(f"Worker {jobid}: open {value} failed ...")
            if not succ:
                continue
            if scp_out:
                scp_out.write(f"{key}\t{ark_out}:{offset}\n")
            done += 1
            if done % 200 == 0:
                print(
                    f"Worker {jobid}: processed {done}/{len(reader)} utterances...",
                    flush=True)
    scp_out.close()
    print(f"Worker {jobid}: archive {done}/{len(reader)} " +
          f"utterances to {ark_out}")


def run(args):
    if args.num_jobs <= 1:
        worker(0, 1, args.wav_scp, args.out_scp, args.out_ark)
    else:
        print(f"Archive audio to .ark files using {args.num_jobs} processes")
        # process fp of out_ark
        ark_out_toks = args.out_ark.split(".")
        ark_prefix = ".".join(ark_out_toks[:-1])
        ark_suffix = ark_out_toks[-1]

        # process fp of out_scp
        scp_out_toks = args.out_scp.split(".")
        scp_prefix = ".".join(scp_out_toks[:-1])
        scp_suffix = scp_out_toks[-1]

        scp_out_list = []
        pool = mp.Pool(args.num_jobs)
        for j in range(args.num_jobs):
            scp_out_list.append(f"{scp_prefix}.{j}.{scp_suffix}")
            packed_args = (j, args.num_jobs, args.wav_scp, scp_out_list[-1],
                           f"{ark_prefix}.{j}.{ark_suffix}")
            pool.apply_async(worker, args=packed_args)
        pool.close()
        pool.join()
        scp_out_list = " ".join(scp_out_list)
        merge_cmd = f"cat {scp_out_list} | sort -k1 > {args.out_scp} && rm {scp_out_list}"
        os.system(merge_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do convert audio to Kaldi's .ark format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input audio script")
    parser.add_argument("out_scp",
                        type=str,
                        help="Output audio script "
                        "with the generated archive files")
    parser.add_argument("out_ark", type=str, help="Output audio archive")
    parser.add_argument("--num-jobs",
                        type=int,
                        default=1,
                        help="Number of the parallel jobs to run")
    args = parser.parse_args()
    run(args)
