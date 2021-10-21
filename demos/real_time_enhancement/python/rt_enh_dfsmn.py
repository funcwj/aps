#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import pathlib

import torch as th
import torch.nn.functional as tf

from aps.io import AudioReader, write_audio
from aps.transform.streaming import StreamingSTFT, StreamingiSTFT
from aps.sse.base import tf_masking
from aps.utils import get_logger, SimpleTimer

# STFT configurations
# ---------------------------------
frame_len = 512
frame_hop = 256
window = "hann"
center = False
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
    lctx = scripted_nnet.lctx
    rctx = scripted_nnet.rctx
    # complex_mask = scripted_nnet.complex_mask
    logger.info(f"lctx = {lctx}, rctx = {rctx}")

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
        num_frames = (num_samples - frame_len) // frame_hop + 1
        pad_samples = (num_frames + rctx - 1) * frame_hop + frame_len
        feat_buffer = None
        stft_buffer = []
        enh = []
        for n in range(0, pad_samples, frame_hop):
            end = n + frame_len
            pad = end - num_samples
            if pad >= frame_len:
                frame = th.zeros(frame_len)
            elif pad > 0 and pad < frame_len:
                frame = tf.pad(wav[n:], (0, pad))
            else:
                frame = wav[n:end]
            # N x F x 2
            frame = forward_stft.step(frame[None, ...])
            # N x F x 1 x 2
            frame = frame[..., None, :]
            stft_buffer.append(frame)
            # N x 1 x F
            frame = transform(frame)
            if feat_buffer is None:
                feat_buffer = [th.zeros_like(frame)] * lctx
            feat_buffer.append(frame)
            if len(feat_buffer) != lctx + rctx + 1:
                continue
            # N x C x F
            chunk = th.cat(feat_buffer, 1)
            # N x F x 1
            masks = scripted_nnet.step(chunk)
            # N x F
            frame = tf_masking(stft_buffer[0], masks)
            frame = frame[..., 0, :]
            enh.append(inverse_stft.step(frame)[0])
            stft_buffer = stft_buffer[1:]
            feat_buffer = feat_buffer[1:]
        last = inverse_stft.flush()
        enh = th.cat(enh + [last], 0)
        enh = enh[center_pad:num_samples - center_pad]
        # for complex mask, we may need re-norm if using si-snr loss
        # if complex_mask:
        #     norm = th.max(th.abs(wav))
        #     enh = enh * norm / th.max(th.abs(enh))
        dur = wav.shape[-1] / args.sr
        time_cost = timer.elapsed() * 60
        logger.info(
            f"Processing utterance {key} done, RTF = {time_cost / dur:.4f}")
        write_audio(dst_dir / f"{key}.wav", enh.numpy())
    logger.info(f"Processed {len(wav_reader)} utterances done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to performance real time speech enhancement using DFSMN model."
        "Please using the script cmd/export_for_libtorch.py to export "
        "TorchScript models and the available real time speech enhancement "
        "models supported by the TorchScript are put under aps/rt_sse/*py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("wav_scp", help="Audio script for evaluation")
    parser.add_argument("scripted_nnet",
                        type=str,
                        help="Scripted DFSMN SE model")
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
