#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import io
import argparse
import subprocess
import multiprocessing as mp

from aps.loader import SegmentReader, write_audio
from aps.utils import get_logger
from kaldi_python_io import Reader as BaseReader

prog_interval = 500
logger = get_logger(__name__)


def worker(jobid, num_jobs, wav_scp, scp_out, ark_out, args):
    scp_out = open(scp_out, "w")
    with open(ark_out, "wb") as wav_ark:
        if args.segment:
            reader = SegmentReader(wav_scp,
                                   args.segment,
                                   norm=False,
                                   sr=args.sr)
        else:
            reader = BaseReader(wav_scp, num_tokens=2, restrict=True)
        done, n = 0, 0
        for key, value in reader:
            n += 1
            if (n - 1) % num_jobs != jobid:
                continue
            wav_ark.write(str.encode(key + " "))
            offset = wav_ark.tell()
            succ = True
            if args.segment:
                io_fd = io.BytesIO()
                write_audio(io_fd,
                            value,
                            sr=args.sr,
                            norm=False,
                            audio_format="wav")
                wav_ark.write(io_fd.getvalue())
            else:
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
                            wav_bytes = wav.read()
                        wav_ark.write(wav_bytes)
                    except FileNotFoundError:
                        succ = False
                        print(f"Worker {jobid}: open {value} failed ...")
            if not succ:
                continue
            if scp_out:
                scp_out.write(f"{key}\t{ark_out}:{offset}\n")
            done += 1
            if done % prog_interval == 0:
                logger.info(
                    f"Worker {jobid}: processed {done}/{len(reader)} utterances..."
                )
    scp_out.close()
    logger.info(f"Worker {jobid}: archive {done}/{len(reader)} " +
                f"utterances to {ark_out}")


def run(args):
    if args.num_jobs <= 1:
        worker(0, 1, args.wav_scp, args.out_scp, args.out_ark, args)
    else:
        num_arks = args.num_arks if args.num_arks >= args.num_jobs else args.num_jobs
        logger.info(f"Archive audio to [0...{args.num_arks}].ark " +
                    f"files using {args.num_jobs} processes")
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
        for j in range(num_arks):
            scp_out_list.append(f"{scp_prefix}.{j}.{scp_suffix}")
            ark_out_part = f"{ark_prefix}.{j}.{ark_suffix}"
            packed_args = (j, num_arks, args.wav_scp, scp_out_list[-1],
                           ark_out_part, args)
            pool.apply_async(worker, args=packed_args)
        pool.close()
        pool.join()
        scp_out_list = " ".join(scp_out_list)
        merge_cmd = f"cat {scp_out_list} | sort -k1 > {args.out_scp} && rm {scp_out_list}"
        logger.info(f"Running bash cmd: {merge_cmd} ...")
        os.system(merge_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do convert audio to Kaldi's .ark format (Linux only)",
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
    parser.add_argument("--num-arks",
                        type=int,
                        default=1,
                        help="Number of the .ark files will created")
    parser.add_argument("--segment",
                        type=str,
                        default="",
                        help="Segment file in Kaldi format")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    args = parser.parse_args()
    run(args)
