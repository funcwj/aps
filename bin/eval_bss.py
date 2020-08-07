#!/usr/bin/env python

# wujian@2020

import pathlib
import argparse

import torch as th
import numpy as np

from aps.loader import WaveReader, write_wav
from aps.utils import get_logger, SimpleTimer
from aps.eval import Computer

logger = get_logger(__name__)


class Separator(Computer):
    """
    Decoder wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        super(Separator, self).__init__(cpt_dir,
                                        device_id=device_id,
                                        task="enh")
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def run(self, src, chunk_len=-1, chunk_hop=-1, mode="time"):
        """
        Args:
            src (Array): (C) x S
        """
        if chunk_hop <= 0 and chunk_len > 0:
            chunk_hop = chunk_len
        N = src.shape[-1]
        src = th.from_numpy(src).to(self.device)
        if chunk_len == -1:
            return self.nnet.infer(src, mode=mode)
        else:
            if mode != "time":
                raise RuntimeError("Now only supports time inference mode")
            chunks = []
            # now only for enhancement task
            for t in range(0, N, chunk_hop):
                pad = N - t - chunk_len
                if pad >= 0:
                    c = src[..., t:t + chunk_len]
                else:
                    # S or P x S
                    if src.dim() == 1:
                        zero = th.zeros(-pad, device=self.device)
                    else:
                        zero = th.zeros(src.shape[0], -pad, device=self.device)
                    c = th.cat([src[..., t:], zero], 0)
                s = self.nnet.infer(c, mode=mode)
                chunks.append(s)
            sep = th.zeros(N)
            for i, c in enumerate(chunks):
                beg = i * chunk_hop
                if i == len(chunks) - 1:
                    sep[beg:] = c[:N - beg]
                else:
                    sep[beg:beg + chunk_len] = c
            return sep


def run(args):
    sep_dir = pathlib.Path(args.sep_dir)
    sep_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(args.checkpoint, device_id=args.device_id)
    mix_reader = WaveReader(args.wav_scp, sr=args.sr, channel=args.channel)

    for key, mix in mix_reader:
        norm = np.max(np.abs(mix))
        timer = SimpleTimer()
        sep = separator.run(mix,
                            chunk_hop=args.chunk_hop,
                            chunk_len=args.chunk_len,
                            mode=args.mode)
        if isinstance(sep, th.Tensor):
            sep = sep.cpu().numpy()
        else:
            sep = np.stack([s.cpu().numpy() for s in sep])
        if args.mode == "time":
            sep = sep * norm / np.max(np.abs(sep))
            # save audio
            write_wav(sep_dir / f"{key}.wav", sep, sr=args.sr)
        else:
            # save TF-mask
            np.save(sep_dir / f"{key}", sep)
        time_cost = timer.elapsed() * 60
        dur = mix.shape[-1] / args.sr
        logger.info(
            f"Processing utterance {key} done, RTF = {time_cost / dur:.2f}")
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
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the separation/enhancement model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--chunk-len",
                        type=int,
                        default=-1,
                        help="Chunk length for inference, "
                        "-1 means the whole utterance")
    parser.add_argument("--chunk-hop",
                        type=int,
                        default=-1,
                        help="Chunk hop size for inference")
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