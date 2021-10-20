#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse

from aps.io import AudioReader
from aps.utils import get_logger

logger = get_logger(__name__)


def run(args):
    wav_reader = AudioReader(args.wav_scp,
                             sr=args.sr,
                             norm=False,
                             failed_if_error=False)
    done, failed = 0, 0
    with open(args.bad_utt, "w") as bad_utt:
        for key, wav in wav_reader:
            done += 1
            if done % 5000 == 0:
                logger.info(f"Tested {done}/{len(wav_reader)} utterances ...")
            if wav is not None:
                continue
            else:
                bad_utt.write(f"{key}\n")
                logger.warning(f"Get bad utterance: {key}")
                failed += 1
    logger.info(f"Tested {len(wav_reader)} utterances done " +
                f"and get {failed} failures ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to check the audio status and output bad utterances list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input audio script")
    parser.add_argument("bad_utt", type=str, help="Output bad utterances list")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    args = parser.parse_args()
    run(args)
