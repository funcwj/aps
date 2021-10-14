#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import io
import argparse
import subprocess
import multiprocessing as mp

from aps.loader import read_audio, write_audio, group_segments
from aps.utils import get_logger
from kaldi_python_io.inst import Writer as BaseWriter
from kaldi_python_io.inst import Reader as BaseReader

prog_interval = 500
logger = get_logger(__name__)


class WaveArkWriter(BaseWriter):
    """
    Write audio stream to archive object
    """

    def __init__(self, scp, ark):
        super(WaveArkWriter, self).__init__(ark, scp)

    def write(self, key, value):
        self.ark_file.write(str.encode(key + " "))
        offset = self.ark_file.tell()
        self.scp_file.write(f"{key}\t{self.ark_path}:{offset}\n")
        self.ark_file.write(value)


def worker(jobid, num_jobs, wav_scp, scp_out, ark_out, args):
    writer = WaveArkWriter(scp_out, ark_out)
    reader = BaseReader(wav_scp, num_tokens=2, restrict=True)
    if args.segment:
        segment = group_segments(args.segment, args.sr)
    else:
        segment = None
    utt_done, n = 0, 0
    num_segs = 0
    # do not load the audio in the outside loop
    for key, value in reader:
        n += 1
        if (n - 1) % num_jobs != jobid:
            continue
        succ = True
        if value[-1] == "|":
            p = subprocess.Popen(value[:-1],
                                 shell=True,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
            [stdout, _] = p.communicate()
            if p.returncode != 0:
                succ = False
                logger.warn(f"Worker {jobid}: running {value[:-1]} failed ...")
            else:
                if segment is None:
                    writer.write(key, stdout)
                    num_segs += 1
                else:
                    if key not in segment:
                        succ = False
                    else:
                        fname = io.BytesIO(stdout)
                        audio = read_audio(fname, norm=False, sr=args.sr)
                        group = segment[key]
                        for info in group:
                            seg_key, beg, end = info
                            io_fd = io.BytesIO()
                            write_audio(io_fd,
                                        audio[..., beg:end],
                                        sr=args.sr,
                                        norm=False,
                                        audio_format="wav")
                            writer.write(seg_key, io_fd.getvalue())
                        num_segs += len(group)
        else:
            try:
                with open(value, "rb") as wav:
                    wav_bytes = wav.read()
                writer.write(key, wav_bytes)
                num_segs += 1
            except FileNotFoundError:
                succ = False
                logger.warn(f"Worker {jobid}: open {value} failed ...")
        if not succ:
            continue
        utt_done += 1
        if utt_done % prog_interval == 0:
            logger.info(
                f"Worker {jobid}: processed {utt_done}/{len(reader)} utterances..."
            )
    logger.info(f"Worker {jobid}: archive {utt_done}/{len(reader)} " +
                f"utterances, {num_segs} segments to {ark_out}")


def run(args):

    if args.num_jobs <= 1:
        worker(0, 1, args.wav_scp, args.out_scp, args.out_ark, args)
    else:
        num_arks = args.num_arks if args.num_arks >= args.num_jobs else args.num_jobs
        logger.info(f"Archive audio to [0...{args.num_arks - 1}].ark " +
                    f"files using {args.num_jobs} processes")

        def prefix_and_suffix(fname):
            toks = fname.split(".")
            return (".".join(toks[:-1]), toks[-1])

        # process fp of out_ark
        ark_prefix, ark_suffix = prefix_and_suffix(args.out_ark)
        # process fp of out_scp
        scp_prefix, scp_suffix = prefix_and_suffix(args.out_scp)

        scp_out_list = []
        pool = mp.Pool(args.num_jobs)
        for n in range(num_arks):
            scp_out_list.append(f"{scp_prefix}.{n}.{scp_suffix}")
            ark_out_part = f"{ark_prefix}.{n}.{ark_suffix}"
            packed_args = (n, num_arks, args.wav_scp, scp_out_list[-1],
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
