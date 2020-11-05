#!/usr/bin/env python

# Copyright 2018 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import tqdm
import argparse
import numpy as np

from aps.loader import AudioReader
from aps.metric.reporter import AverageReporter
from aps.metric.sse import permute_metric

from kaldi_python_io import Reader as BaseReader


def run(args):
    splited_est_scps = args.est_scp.split(",")
    splited_ref_scps = args.ref_scp.split(",")
    if len(splited_ref_scps) != len(splited_est_scps):
        raise RuntimeError(f"Number of the speakers doesn't matched")
    single_speaker = len(splited_est_scps) == 1

    reporter = AverageReporter(args.spk2class,
                               name=args.metric.upper(),
                               unit="dB")
    utt_val = open(args.per_utt, "w") if args.per_utt else None
    utt_ali = open(args.utt_ali, "w") if args.utt_ali else None

    if single_speaker:
        est_reader = AudioReader(args.est_scp, sr=args.sr)
        ref_reader = AudioReader(args.ref_scp, sr=args.sr)
        for key, sep in tqdm.tqdm(est_reader):
            ref = ref_reader[key]
            end = min(sep.size, ref.size)
            metric = permute_metric(args.metric,
                                    ref[:end],
                                    sep[:end],
                                    fs=args.sr,
                                    compute_permutation=False)
            reporter.add(key, metric)
            if utt_val:
                utt_val.write(f"{key}\t{metric:.2f}\n")
    else:
        est_reader = [AudioReader(scp, sr=args.sr) for scp in splited_est_scps]
        ref_reader = [AudioReader(scp, sr=args.sr) for scp in splited_ref_scps]
        main_reader = est_reader[0]

        for key in tqdm.tqdm(main_reader.index_keys):
            est = np.stack([reader[key] for reader in est_reader])
            ref = np.stack([reader[key] for reader in ref_reader])
            end = min(est.shape[-1], ref.shape[-1])
            metric, ali = permute_metric(args.metric,
                                         ref[:, :end],
                                         est[:, :end],
                                         fs=args.sr,
                                         compute_permutation=True)
            reporter.add(key, metric)
            if utt_val:
                utt_val.write(f"{key}\t{metric:.2f}\n")
            if utt_ali:
                ali_str = " ".join(map(str, ali))
                utt_ali.write(f"{key}\t{ali_str}\n")
    reporter.report()
    if utt_val:
        utt_val.close()
    if utt_ali:
        utt_ali.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute the audio metrics to measure the quality "
        "of the speech separation & enhancement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("est_scp",
                        type=str,
                        help="Estimated speech scripts, waiting for measure"
                        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument("ref_scp",
                        type=str,
                        help="Reference speech scripts, as ground truth for"
                        " Si-SDR computation")
    parser.add_argument("--metric",
                        type=str,
                        required=True,
                        choices=["sdr", "sisnr", "pesq", "stoi"],
                        help="Name of the audio metric")
    parser.add_argument("--spk2class",
                        type=str,
                        default="",
                        help="If assigned, report results "
                        "per class (gender or degree)")
    parser.add_argument("--per-utt",
                        type=str,
                        default="",
                        help="If assigned, report snr "
                        "improvement for each utterance")
    parser.add_argument("--utt-ali",
                        type=str,
                        default="",
                        help="If assigned, output audio alignments")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    args = parser.parse_args()
    run(args)
