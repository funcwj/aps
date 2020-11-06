#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import argparse

import torch as th

from aps.loader import AudioReader
from aps.libs import aps_transform


def run(args):
    with open(args.conf, "r") as f:
        conf = yaml.full_load(f)
    trans_key = f"{args.transform}_transform"
    if trans_key not in conf:
        print(f"No {trans_key} in {args.conf}, exist ...")
    wav_reader = AudioReader(args.wav_scp, sr=args.sr, channel=args.channel)

    feats_conf_list = conf[trans_key]["feats"].split("-")
    for remove_key in ["cmvn", "splice", "aug", "delta"]:
        if remove_key in feats_conf_list:
            feats_conf_list.remove(remove_key)
            print(f"Removed key in feature configuration: {remove_key}")

    feats_conf = "-".join(feats_conf_list)
    conf[trans_key]["feats"] = feats_conf
    transform = aps_transform(args.transform)(**conf[trans_key])
    print(f"Compute gmvn on feature {feats_conf}")
    gmvn = th.zeros([2, transform.feats_dim])
    num_utts = 0
    for _, wav in wav_reader:
        # 1 x C x T x F
        feats = transform(th.from_numpy(wav[None, ...]), None)[0]
        if feats.dim() == 3:
            num_utts += 1
            gmvn[0] += th.mean(feats[0], 0)
            gmvn[1] += th.mean(feats[0]**2, 0)
        else:
            num_utts += feats.shape[0]
            for c in range(feats.shape[0]):
                gmvn[0] += th.mean(feats[0, c], 0)
                gmvn[1] += th.mean(feats[0, c]**2, 0)
        if num_utts % 500 == 0:
            print(f"Processed {num_utts} utterances...")
    gmvn[0] = gmvn[0] / num_utts
    gmvn[1] = (gmvn[1] / num_utts - gmvn[0])**0.5
    if th.sum(th.isnan(gmvn)):
        raise RuntimeError("Got NAN in gmvn, please check")
    print(f"Global mean/variance:\n{gmvn}")
    print("Save global mean/variance to " +
          f"{args.out_mvn} over {num_utts} utterances")
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
    args = parser.parse_args()
    run(args)
