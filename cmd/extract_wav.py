#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import pathlib

from aps.loader import AudioReader, SegmentReader, write_audio
from aps.utils import get_logger

logger = get_logger(__name__)


def run(args):
    if args.segment:
        wav_reader = SegmentReader(args.wav_scp,
                                   args.segment,
                                   sr=args.sr,
                                   norm=False,
                                   channel=args.channel)
    else:
        wav_reader = AudioReader(args.wav_scp,
                                 sr=args.sr,
                                 norm=False,
                                 channel=args.channel)
    dump_dir = pathlib.Path(args.wav_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for key, wav in wav_reader:
        write_audio(dump_dir / f"{key}.wav", wav, sr=args.sr, norm=False)
        done += 1
        if done % 200 == 0:
            logger.info(f"Extracted {done} utterances done ...")
    logger.info(f"Extract {done} to {dump_dir} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do convert audio to Kaldi's .ark format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", type=str, help="Input audio script")
    parser.add_argument("wav_dir", type=str, help="Output directory")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the audio")
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index of the audio")
    parser.add_argument("--segment",
                        type=str,
                        default="",
                        help="Segment file in Kaldi format")
    args = parser.parse_args()
    run(args)
