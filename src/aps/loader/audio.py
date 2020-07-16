#!/usr/bin/env python

# wujian@2019

import os
import soundfile as sf
import numpy as np
import scipy.signal as ss

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


def read_wav(fname, beg=0, end=None, norm=True, sr=16000):
    """
    Read wave files using soundfile (support multi-channel & chunk)
    args:
        fname: file name or object
        beg, end: begin and end index for chunk-level reading
        norm: normalized samples between -1 and 1
        sr: sample rate of the audio
    return:
        samps: in shape C x N
        sr: sample rate
    """
    # samps: N x C or N
    #   N: number of samples
    #   C: number of channels
    samps, ret_sr = sf.read(fname,
                            start=beg,
                            stop=end,
                            dtype="float32" if norm else "int16")
    if sr != ret_sr:
        raise RuntimeError(f"Expect sr={sr} of {fname}, get {ret_sr} instead")
    if not norm:
        samps = samps.astype("float32")
    # put channel axis first
    # N x C => C x N
    if samps.ndim != 1:
        samps = np.transpose(samps)
    return samps


def write_wav(fname, samps, sr=16000, norm=True):
    """
    Write wav files, support single/multi-channel
    """
    samps = samps.astype("float32" if norm else "int16")
    # scipy.io.wavfile/soundfile could write single/multi-channel files
    # for multi-channel, accept ndarray [Nsamples, Nchannels]
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # make dirs
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    # NOTE: librosa 0.6.0 seems could not write non-float narray
    #       so use scipy.io.wavfile/soundfile instead
    # wf.write(fname, sr, samps_int16)
    sf.write(fname, samps, sr)


def add_room_response(spk, rir, early_energy=False, sr=16000):
    """
    Convolute source signal with selected rirs
    Args
        spk: S
        rir: N x R
    Return
        revb: N x S
    """
    if spk.ndim != 1:
        raise RuntimeError(f"Can not convolve rir with {spk.ndim}D signals")
    S = spk.shape[-1]
    revb = ss.convolve(spk[None, ...], rir)[..., :S]
    revb = np.asarray(revb)

    if early_energy:
        rir_ch0 = rir[0]
        rir_peak = np.argmax(rir_ch0)
        rir_beg_idx = max(0, int(rir_peak - 0.001 * sr))
        rir_end_idx = min(rir_ch0.size, int(rir_peak + 0.05 * sr))
        early_rir = np.zeros_like(rir_ch0)
        early_rir[rir_beg_idx:rir_end_idx] = rir_ch0[rir_beg_idx:rir_end_idx]
        early_rev = ss.convolve(spk, early_rir)[:S]
        return revb, np.mean(early_rev**2)
    else:
        return revb, np.mean(revb[0]**2)