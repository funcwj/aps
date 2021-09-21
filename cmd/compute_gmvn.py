#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import os
import yaml
import copy
import argparse

import torch as th
import multiprocessing as mp

from aps.loader import AudioReader
from aps.libs import aps_transform
from aps.utils import get_logger

key_to_remove = ["perturb", "cmvn", "splice", "aug", "delta"]

logger = get_logger(__name__)
prog_interval = 500


def worker(jobid, transform, args, stats):
    wav_reader = AudioReader(args.wav_scp, sr=args.sr, channel=args.channel)
    gmvn = th.zeros([2, transform.feats_dim])
    n, done, num_frames = 0, 0, 0
    for key, wav in wav_reader:
        n += 1
        if (n - 1) % args.num_jobs != jobid:
            continue
        # 1 x C x T x F
        feats = transform(th.from_numpy(wav[None, ...]), None)[0]
        utt_frames = feats.shape[-2]
        if feats.dim() == 3:
            num_frames += utt_frames
            gmvn[0] += th.sum(feats[0], 0)
            gmvn[1] += th.sum(feats[0]**2, 0)
        else:
            num_channels = feats.shape[1]
            num_frames += utt_frames * num_channels
            for c in range(num_channels):
                gmvn[0] += th.sum(feats[0, c], 0)
                gmvn[1] += th.sum(feats[0, c]**2, 0)
        done += 1
        if done % prog_interval == 0:
            logger.info(f"Worker {jobid}: processed " +
                        f"{done}/{len(wav_reader)} utterances...")
    state_dict = {"num_frames": num_frames, "gmvn": gmvn}
    th.save(state_dict, stats)
    logger.info(
        f"Worker {jobid}: processed {done}/{len(wav_reader)} utterances done")


def run(args):
    # important for multiprocessing
    th.set_num_threads(1)
    logger.info("Compute global mean/variance normalization " +
                f"using {args.num_jobs} processes")
    with open(args.conf, "r") as f:
        conf = yaml.full_load(f)
    trans_key = f"{args.transform}_transform"
    if trans_key not in conf:
        logger.info(f"No {trans_key} in {args.conf}, exist ...")

    feats_conf_list = conf[trans_key]["feats"].split("-")
    for key in key_to_remove:
        if key in feats_conf_list:
            feats_conf_list.remove(key)
            logger.info(f"Removed key in feature configuration: {key}")

    feats_conf = "-".join(feats_conf_list)
    conf[trans_key]["feats"] = feats_conf
    transform = aps_transform(args.transform)(**conf[trans_key])
    transform.eval()
    logger.info(f"Compute gmvn on feature {feats_conf}")

    stats_list = []
    ps = []
    for j in range(args.num_jobs):
        stats_list.append(args.out_mvn + f".{j}.stats.pt")
        packed_args = [j, copy.deepcopy(transform), args, stats_list[-1]]
        p = mp.Process(target=worker, args=packed_args)
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    gmvn = th.zeros([2, transform.feats_dim])
    num_frames = 0
    for stats in stats_list:
        s = th.load(stats)
        num_frames += s["num_frames"]
        gmvn += s["gmvn"]
    gmvn[0] = gmvn[0] / num_frames
    gmvn[1] = (gmvn[1] / num_frames - gmvn[0])**0.5
    if th.sum(th.isnan(gmvn)):
        raise RuntimeError("Got NAN in gmvn, please check")
    stats_list = " ".join(stats_list)
    logger.info(f"Removing {stats_list} ...")
    os.system(f"rm {stats_list}")
    logger.info(f"Global mean/variance:\n{gmvn}")
    logger.info("Save global mean/variance to " +
                f"{args.out_mvn} over {num_frames} frames")
    th.save(gmvn, args.out_mvn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute global mean & variance normalization statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp",
                        type=str,
                        help="Audio script for global "
                        "mean and variance computation")
    parser.add_argument("conf", type=str, help="Training configuration")
    parser.add_argument("out_mvn", type=str, help="Output cmvn object")
    parser.add_argument("--transform",
                        default="asr",
                        choices=["asr", "enh"],
                        help="Using asr_transform or enh_transform")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    parser.add_argument("--channel",
                        default=-1,
                        type=int,
                        help="Which channel to use (for multi-channel setups)")
    parser.add_argument("--num-jobs",
                        default=1,
                        type=int,
                        help="Number of jobs to run in parallel")
    args = parser.parse_args()
    run(args)
