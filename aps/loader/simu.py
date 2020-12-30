#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Adopt from my another project: https://github.com/funcwj/setk
See https://github.com/funcwj/setk/tree/master/doc/data_simu for command line usage
"""
import argparse
import numpy as np

from aps.loader.audio import read_audio, add_room_response
from aps.opts import StrToBoolAction
from aps.const import EPSILON


def coeff_snr(sig_pow, ref_pow, snr):
    """
    For
        mix = Sa + alpha*Sb
    Given
        SNR = 10*log10[Pa/(Pb * alpha^2)]
    we got
        alpha = Pa/[Pb*10^(SNR/10)]^0.5
    """
    return (ref_pow / (sig_pow * 10**(snr / 10) + EPSILON))**0.5


def add_speaker(mix_nsamps,
                src_spk,
                src_begin,
                sdr,
                src_rir=None,
                channel=-1,
                sr=16000):
    """
    Mix source speakers
    """
    spk_image, spk_power = [], []
    for i, spk in enumerate(src_spk):
        if src_rir is None:
            src = spk[None, ...] if spk.ndim == 1 else spk
            spk_image.append(src)
            spk_power.append(np.mean(src[0]**2))
        else:
            rir = src_rir[i]
            if rir.ndim == 1:
                rir = rir[None, ...]
            if channel >= 0:
                if rir.ndim == 2:
                    rir = rir[channel:channel + 1]
            revb, p = add_room_response(spk, rir, sr=sr)
            spk_image.append(revb)
            spk_power.append(p)
    # make mix
    N, _ = spk_image[0].shape
    mix = [np.zeros([N, mix_nsamps], dtype=np.float32) for _ in src_spk]
    # start mixing
    ref_power = spk_power[0]
    for i, image in enumerate(spk_image):
        dur = image.shape[-1]
        beg = src_begin[i]
        coeff = 1 if i == 0 else coeff_snr(spk_power[i], ref_power, sdr[i])
        mix[i][..., beg:beg + dur] += coeff * image
    return mix


def add_point_noise(mix_nsamps,
                    ref_power,
                    noise,
                    noise_begin,
                    snr,
                    noise_rir=None,
                    channel=-1,
                    repeat=False,
                    sr=16000):
    """
    Add pointsource noises
    """
    image = []
    image_power = []
    for i, noise in enumerate(noise):
        beg = noise_begin[i]
        if not repeat:
            dur = min(noise.shape[-1], mix_nsamps - beg)
        else:
            dur = mix_nsamps - beg
            # if short, then padding
            if noise.shape[-1] < dur:
                noise = np.pad(noise, (0, dur - noise.shape[-1]), mode="wrap")

        if noise_rir is None:
            src = noise[None, ...] if noise.ndim == 1 else noise

            image.append(src)
            image_power.append(np.mean(src[0, :dur]**2) if dur > 0 else 0)
        else:
            rir = noise_rir[i]
            if rir.ndim == 1:
                rir = rir[None, ...]
            if channel >= 0:
                if rir.ndim == 2:
                    rir = rir[channel:channel + 1]
            revb, revb_power = add_room_response(noise[:dur], rir, sr=sr)
            image.append(revb)
            image_power.append(revb_power)
    # make noise mix
    N, _ = image[0].shape
    mix = np.zeros([N, mix_nsamps], dtype=np.float32)
    # start mixing
    for i, img in enumerate(image):
        beg = noise_begin[i]
        coeff = coeff_snr(image_power[i], ref_power, snr[i])
        mix[..., beg:beg + dur] += coeff * img[..., :dur]
    return mix


def load_audio(src_args, beg=None, end=None, sr=16000):
    """
    Load audio from args.xxx
    """
    if src_args:
        src_path = src_args.split(",")
        beg_int = [None for _ in src_path]
        end_int = [None for _ in src_path]
        if beg:
            beg_int = [int(v) for v in beg.split(",")]
        if end:
            end_int = [int(v) for v in end.split(",")]
        return [
            read_audio(s, sr=sr, beg=b, end=e)
            for s, b, e in zip(src_path, beg_int, end_int)
        ]
    else:
        return None


def run_simu(args):

    def arg_float(src_args):
        return [float(s) for s in src_args.split(",")] if src_args else None

    src_spk = load_audio(args.src_spk, sr=args.sr)
    src_rir = load_audio(args.src_rir, sr=args.sr)
    if src_rir:
        if len(src_rir) != len(src_spk):
            raise RuntimeError(
                f"Number of --src-rir={args.src_rir} do not match with " +
                f"--src-spk={args.src_spk} option")
    sdr = arg_float(args.src_sdr)
    if len(src_spk) > 1 and not sdr:
        raise RuntimeError("--src-sdr need to be assigned for " +
                           f"--src-spk={args.src_spk}")
    if sdr:
        if len(src_spk) - 1 != len(sdr):
            raise RuntimeError("Number of --src-snr - 1 do not match with " +
                               "--src-snr option")
        sdr = [0] + sdr

    src_begin = arg_float(args.src_begin)
    if src_begin:
        src_begin = [int(v) for v in src_begin]
    else:
        src_begin = [0 for _ in src_spk]

    # number samples of the mixture
    mix_nsamps = max([b + s.size for b, s in zip(src_begin, src_spk)])

    point_noise_rir = load_audio(args.point_noise_rir, sr=args.sr)

    point_noise_end = [
        str(int(v) + mix_nsamps) for v in args.point_noise_offset.split()
    ]
    point_noise = load_audio(args.point_noise,
                             beg=args.point_noise_offset,
                             end=",".join(point_noise_end),
                             sr=args.sr)

    if args.point_noise:
        if point_noise_rir:
            if len(point_noise) != len(point_noise_rir):
                raise RuntimeError(
                    f"Number of --point-noise-rir={args.point_noise_rir} do not match with "
                    + f"--point-noise={args.point_noise} option")
        point_snr = arg_float(args.point_noise_snr)
        if not point_snr:
            raise RuntimeError("--point-noise-snr need to be assigned for " +
                               f"--point-noise={args.point_noise}")
        if len(point_noise) != len(point_snr):
            raise RuntimeError(
                f"Number of --point-noise-snr={args.point_noise_snr} do not match with "
                + f"--point-noise={args.point_noise} option")

        point_begin = arg_float(args.point_noise_begin)
        if point_begin:
            point_begin = [int(v) for v in point_begin]
        else:
            point_begin = [0 for _ in point_noise]

    isotropic_noise = load_audio(args.isotropic_noise,
                                 beg=str(args.isotropic_noise_offset),
                                 end=str(args.isotropic_noise_offset +
                                         mix_nsamps),
                                 sr=args.sr)
    if isotropic_noise:
        isotropic_noise = isotropic_noise[0]
        isotropic_snr = arg_float(args.isotropic_noise_snr)
        if not isotropic_snr:
            raise RuntimeError(
                "--isotropic-snr need to be assigned for " +
                f"--isotropic-noise={args.isotropic_noise} option")
        isotropic_snr = isotropic_snr[0]
    else:
        isotropic_snr = None

    # add speakers
    spk = add_speaker(mix_nsamps,
                      src_spk,
                      src_begin,
                      sdr,
                      src_rir=src_rir,
                      channel=args.dump_channel,
                      sr=args.sr)
    spk_utt = sum(spk)
    mix = spk_utt.copy()

    spk_power = np.mean(spk_utt[0]**2)
    if point_noise:
        noise = add_point_noise(mix_nsamps,
                                spk_power,
                                point_noise,
                                point_begin,
                                point_snr,
                                noise_rir=point_noise_rir,
                                channel=args.dump_channel,
                                repeat=args.point_noise_repeat,
                                sr=args.sr)
        num_channels = spk_utt.shape[0]
        if num_channels != noise.shape[0]:
            if num_channels == 1:
                noise = noise[0:1]
            else:
                raise RuntimeError("Channel mismatch between source speaker " +
                                   "configuration and pointsource noise's, " +
                                   f"{num_channels} vs {noise.shape[0]}")
        mix = spk_utt + noise
    else:
        noise = None

    ch = args.dump_channel
    if isotropic_noise is not None:
        N, _ = spk_utt.shape
        if N == 1:
            if isotropic_noise.ndim == 1:
                isotropic_noise = isotropic_noise[None, ...]
            else:
                if ch >= 0:
                    isotropic_noise = isotropic_noise[ch:ch + 1]
                else:
                    raise RuntimeError(
                        "Single channel mixture vs multi-channel "
                        "isotropic noise")
        else:
            if isotropic_noise.shape[0] != N:
                raise RuntimeError(
                    "Channel number mismatch between mixture and isotropic noise, "
                    + f"{N} vs {isotropic_noise.shape[0]}")

        dur = min(mix_nsamps, isotropic_noise.shape[-1])
        isotropic_chunk = isotropic_noise[0, :dur]
        power = np.mean(isotropic_chunk**2)
        coeff = coeff_snr(power, spk_power, isotropic_snr)
        mix[..., :dur] += coeff * isotropic_chunk

        if noise is None:
            noise = coeff * isotropic_chunk
        else:
            noise[..., :dur] += coeff * isotropic_chunk

    factor = args.norm_factor / (np.max(np.abs(mix)) + EPSILON)

    mix = mix.squeeze() * factor
    spk = [s[0] * factor for s in spk]

    if noise is None:
        return mix, spk, None
    else:
        return mix, spk, noise[0] * factor


def make_argparse():
    parser = argparse.ArgumentParser(
        description="Command to do audio data simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-spk",
                        type=str,
                        required=True,
                        help="Source speakers, e.g., spk1.wav,spk2.wav")
    parser.add_argument("--src-rir",
                        type=str,
                        default="",
                        help="RIRs for each source speakers")
    parser.add_argument("--src-sdr",
                        type=str,
                        default="",
                        help="SDR for each speakers (if needed)")
    parser.add_argument("--src-begin",
                        type=str,
                        default="",
                        help="Begining samples on the mixture utterances")
    parser.add_argument("--point-noise",
                        type=str,
                        default="",
                        help="Add pointsource noises")
    parser.add_argument("--point-noise-rir",
                        type=str,
                        default="",
                        help="RIRs of the pointsource noises (if needed)")
    parser.add_argument("--point-noise-snr",
                        type=str,
                        default="",
                        help="SNR of the pointsource noises")
    parser.add_argument("--point-noise-begin",
                        type=str,
                        default="",
                        help="Begining samples of the "
                        "pointsource noises on the mixture "
                        "utterances (if needed)")
    parser.add_argument("--point-noise-offset",
                        type=str,
                        default="",
                        help="Add from the offset position "
                        "of the pointsource noise")
    parser.add_argument("--point-noise-repeat",
                        action=StrToBoolAction,
                        default="false",
                        help="Repeat the pointsource noise or not")
    parser.add_argument("--isotropic-noise",
                        type=str,
                        default="",
                        help="Add isotropic noises")
    parser.add_argument("--isotropic-noise-snr",
                        type=str,
                        default="",
                        help="SNR of the isotropic noises")
    parser.add_argument("--isotropic-noise-offset",
                        type=int,
                        default=0,
                        help="Add noise from the offset position "
                        "of the isotropic noise")
    parser.add_argument("--dump-channel",
                        type=int,
                        default=-1,
                        help="Index of the channel to dump out (-1 means all)")
    parser.add_argument('--norm-factor',
                        type=float,
                        default=0.9,
                        help="Normalization factor of the final output")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Value of the sample rate")
    return parser
