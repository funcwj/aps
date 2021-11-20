#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pathlib
import argparse

import torch as th
import numpy as np

from aps.io import AudioReader, write_audio
from aps.utils import get_logger, SimpleTimer
from aps.eval import NnetEvaluator, ChunkStitcher

logger = get_logger(__name__)
logger_interval = 30


class Separator(NnetEvaluator):
    """
    Decoder wrapper
    """

    def __init__(self,
                 cpt_dir,
                 cpt_tag: str = "best",
                 sr: int = 16000,
                 device_id: int = -1,
                 chunk_cfg: str = "0,-1,0") -> None:
        super(Separator, self).__init__(cpt_dir,
                                        cpt_tag=cpt_tag,
                                        device_id=device_id)
        lctx, chunk_len, rctx = [
            int(v * sr) for v in list(map(float, chunk_cfg.split(",")))
        ]
        if chunk_len > 0:
            logger.info(
                f"Perform chunk-wise evaluation: length = {chunk_len}, " +
                f"lctx = {lctx}, rctx = {rctx}")
            self.stitcher = ChunkStitcher(chunk_len, lctx, rctx)
        else:
            self.stitcher = None
        self.chunk_hop = chunk_len
        self.chunk_len = chunk_len + rctx
        self.lctx = lctx

    def run(self, src: np.ndarray, mode: str = "time") -> th.Tensor:
        """
        Args:
            src (ndarray): (C) x S
        """
        expected_length = src.shape[-1]
        src = th.from_numpy(src).to(self.device)
        if self.stitcher is None:
            return self.nnet.infer(src, mode=mode)
        else:
            if mode != "time":
                raise RuntimeError("Now only supports time inference mode")
            chunks = []
            beg = self.lctx
            # for beg in range(0, expected_length, self.chunk_hop):
            while True:
                if (beg - self.lctx) % (logger_interval * self.chunk_hop) == 0:
                    progress = beg * 100 / expected_length
                    logger.info(
                        f"--- Processing chunks, done {progress:.2f}% ...")
                pad = expected_length - beg - self.chunk_len
                if pad < 0:
                    # last chunk, need padding
                    if src.dim() == 1:
                        zero = th.zeros(-pad, device=self.device)
                    else:
                        zero = th.zeros(src.shape[0], -pad, device=self.device)
                    mix_chunk = th.cat([src[..., beg - self.lctx:], zero], 0)
                else:
                    mix_chunk = src[..., beg - self.lctx:beg + self.chunk_len]
                sep_chunk = self.nnet.infer(mix_chunk, mode=mode)
                if isinstance(sep_chunk, th.Tensor):
                    sep_chunk = sep_chunk.cpu()
                else:
                    sep_chunk = [s.cpu() for s in sep_chunk]
                chunks.append(sep_chunk)
                beg += self.chunk_hop
                if pad < 0:
                    break
            logger.info("--- Stitch & Reorder ...")
            return self.stitcher.stitch(chunks, expected_length)


def run(args):
    sep_dir = pathlib.Path(args.sep_dir)
    sep_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(args.checkpoint,
                          cpt_tag=args.tag,
                          device_id=args.device_id,
                          chunk_cfg=args.chunk_cfg)
    mix_reader = AudioReader(args.wav_scp, sr=args.sr, channel=args.channel)

    done = 0
    for key, mix in mix_reader:
        timer = SimpleTimer()
        norm = np.max(np.abs(mix))
        sep = separator.run(mix, mode=args.mode)
        if isinstance(sep, th.Tensor):
            sep = sep.cpu().numpy()
        else:
            sep = np.stack([s.cpu().numpy() for s in sep])
        if args.mode == "time":
            sep = sep * norm / np.max(np.abs(sep))
            # save audio
            write_audio(sep_dir / f"{key}.wav", sep, sr=args.sr)
        else:
            # save TF-mask
            np.save(sep_dir / f"{key}", sep)
        time_cost = timer.elapsed() * 60
        dur = mix.shape[-1] / args.sr
        done += 1
        logger.info(
            f"Processing utterance {key} done ({done}/{len(mix_reader)}), "
            f"RTF = {time_cost / dur:.4f}")
    logger.info(f"Processed {len(mix_reader)} utterances done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to do blind speech separation (enhancement)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp",
                        type=str,
                        help="Mixture & Noisy input audio scripts")
    parser.add_argument("sep_dir",
                        type=str,
                        help="Directory to dump enhanced/separated output")
    parser.add_argument("--mode",
                        type=str,
                        choices=["time", "freq"],
                        default="time",
                        help="Inference mode of the bss model")
    parser.add_argument("--tag",
                        type=str,
                        default="best",
                        help="Tag name to load the checkpoint: (tag).pt.tar")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the separation/enhancement model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--chunk-cfg",
                        type=str,
                        default="0,-1,0",
                        help="Configurations for chunk-wise processing "
                        "(left context & chunk size & right context in "
                        "seconds)")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the source audio")
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index for source audio")
    args = parser.parse_args()
    run(args)
