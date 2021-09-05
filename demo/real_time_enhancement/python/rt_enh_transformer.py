#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import pathlib

import torch as th
import torch.nn.functional as tf

from aps.loader.audio import AudioReader, write_audio
from aps.transform.streaming import StreamingSTFT, StreamingiSTFT
from aps.sse.base import tf_masking
from aps.utils import get_logger, SimpleTimer

# STFT configurations
# ---------------------------------
frame_len = 512
frame_hop = 256
window = "hann"
center = True
# ---------------------------------

logger = get_logger(__name__)


def run(args):
    # disable grad
    th.set_grad_enabled(False)

    dst_dir = pathlib.Path(args.dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)
    wav_reader = AudioReader(args.wav_scp, channel=args.channel, sr=args.sr)
    scripted_nnet = th.jit.load(args.scripted_nnet)
    scripted_nnet.eval()
    transform = th.jit.load(args.transform)
    transform.eval()

    # Get attributes
    chunk = scripted_nnet.chunk
    complex_mask = scripted_nnet.complex_mask

    forward_stft = StreamingSTFT(frame_len,
                                 frame_hop,
                                 window=window,
                                 center=False)
    inverse_stft = StreamingiSTFT(frame_len,
                                  frame_hop,
                                  window=window,
                                  center=False)
    for key, wav in wav_reader:
        wav = th.from_numpy(wav)
        center_pad = frame_len // 2 if center else 0
        wav = tf.pad(wav, (center_pad, center_pad))
        # reset status
        scripted_nnet.reset()
        timer = SimpleTimer()

        num_samples = wav.shape[-1]
        wav_chunk_size = (chunk - 1) * frame_hop + frame_len
        enh = []
        for n in range(0, num_samples, wav_chunk_size):
            end = n + wav_chunk_size
            pad = end - num_samples
            if pad > 0:
                wav_chunk = tf.pad(wav[n:], (0, pad))
            else:
                wav_chunk = wav[n:end]
            stft_chunk = []
            # STFT
            for t in range(chunk):
                frame = wav_chunk[t * frame_hop:t * frame_hop + frame_len]
                stft_chunk.append(forward_stft.step(frame[None, ...]))
            # STFT: N x F x C x 2
            stft_chunk = th.stack(stft_chunk, -2)
            # feature: N x T x F
            feats = transform(stft_chunk)
            # N x F x C x 2 (complex) or N x F x C (real)
            masks = scripted_nnet.step(feats)
            # N x F x C x 2
            stft_chunk = tf_masking(stft_chunk, masks)
            # iSTFT
            for t in range(chunk):
                frame = stft_chunk[..., t, :]
                enh.append(inverse_stft.step(frame)[0])
        last = inverse_stft.flush()
        enh = th.cat(enh + [last], 0)
        enh = enh[center_pad:num_samples - center_pad]
        # for complex mask, we may need re-norm
        # if complex_mask:
        #     norm = th.max(th.abs(wav))
        #     enh = enh * norm / th.max(th.abs(enh))
        dur = wav.shape[-1] / args.sr
        time_cost = timer.elapsed()
        logger.info(
            f"Processing utterance {key} done, RTF = {time_cost / dur:.4f}")
        write_audio(dst_dir / f"{key}.wav", enh.numpy())
    logger.info(f"Processed {len(wav_reader)} utterances done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to evaluate transformer based real time speech enhancement",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", help="Audio script for evaluation")
    parser.add_argument("scripted_nnet",
                        type=str,
                        help="Scripted transformer based SE model")
    parser.add_argument("--transform",
                        type=str,
                        required=True,
                        help="Scripted feature transform network")
    parser.add_argument("dst_dir",
                        type=str,
                        help="Path to save the enhanced audio")
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
