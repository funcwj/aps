#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import io
import os
import subprocess

import numpy as np
import soundfile as sf
import scipy.signal as ss

from kaldi_python_io import Reader as BaseReader
from typing import Optional, IO, Union, Any, NoReturn, Tuple


def read_audio(fname: Union[str, IO[Any]],
               beg: int = 0,
               end: Optional[int] = None,
               norm: bool = True,
               sr: int = 16000) -> np.ndarray:
    """
    Read audio files using soundfile (support multi-channel & chunk)
    Args:
        fname: file name or object
        beg, end: begin and end index for chunk-level reading
        norm: normalized samples between -1 and 1
        sr: sample rate of the audio
    Return:
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


def write_audio(fname: Union[str, IO[Any]],
                samps: np.ndarray,
                sr: int = 16000,
                norm: bool = True) -> NoReturn:
    """
    Write audio files, support single/multi-channel
    Args:
        fname: IO object or str
        samps: np.ndarray, C x S or S
        sr: sample rate
        norm: keep same as the one in read_audio
    """
    samps = samps.astype("float32" if norm else "int16")
    # for multi-channel, accept ndarray N x C
    if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
        samps = np.transpose(samps)
        samps = np.squeeze(samps)
    # make dirs
    if isinstance(fname, str):
        parent = os.path.dirname(fname)
        if parent and not os.path.exists(parent):
            os.makedirs(parent)
    sf.write(fname, samps, sr)


def add_room_response(spk: np.ndarray,
                      rir: np.ndarray,
                      early_energy: bool = False,
                      sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Convolute source signal with selected rirs
    Args
        spk: S, close talk signal
        rir: N x R, single or multi-channel RIRs
        early_energy: return energy of early parts
        sr: sample rate of the signal
    Return
        revb: N x S, reverberated signals
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


def run_command(command: str, wait: bool = True):
    """
    Runs shell commands
    """
    p = subprocess.Popen(command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode != 0:
            stderr_str = bytes.decode(stderr)
            raise Exception("There was an error while running the " +
                            f"command \"{command}\":\n{stderr_str}\n")
        return stdout, stderr
    else:
        return p


class AudioReader(BaseReader):
    """
    Sequential/Random Reader for single/multiple channel audio using soundfile as the backend
    The format of wav.scp follows Kaldi's definition:
        key1 /path/to/key1.wav
        key2 /path/to/key2.wav
        ...
    or
        key1 sox /home/data/key1.wav -t wav - remix 1 |
        key2 sox /home/data/key2.wav -t wav - remix 1 |
        ...
    or
        key1 /path/to/ark1:XXXX
        key2 /path/to/ark1:XXXY
    are supported

    Args:
        wav_scp: path of the audio script
        sr: sample rate of the audio
        norm: normalize audio samples between (-1, 1) if true
        channel: read audio at #channel if > 0 (-1 means all)
    """

    def __init__(self,
                 wav_scp: str,
                 sr: int = 16000,
                 norm: bool = True,
                 channel: int = -1) -> None:
        super(AudioReader, self).__init__(wav_scp, num_tokens=2)
        self.sr = sr
        self.ch = channel
        self.norm = norm
        self.mngr = {}

    def _load(self, key: str) -> Optional[np.ndarray]:
        fname = self.index_dict[key]
        samps = None
        # return C x N or N
        if ":" in fname:
            tokens = fname.split(":")
            if len(tokens) != 2:
                raise RuntimeError(f"Value format error: {fname}")
            fname, offset = tokens[0], int(tokens[1])
            # get ark object
            if fname not in self.mngr:
                self.mngr[fname] = open(fname, "rb")
            wav_ark = self.mngr[fname]
            # wav_ark = open(fname, "rb")
            # seek and read
            wav_ark.seek(offset)
            try:
                samps = read_audio(wav_ark, norm=self.norm, sr=self.sr)
            except RuntimeError:
                print(f"Read audio {key} {fname}:{offset} failed...",
                      flush=True)
        else:
            if fname[-1] == "|":
                shell, _ = run_command(fname[:-1], wait=True)
                fname = io.BytesIO(shell)
            try:
                samps = read_audio(fname, norm=self.norm, sr=self.sr)
            except RuntimeError:
                print(f"Load audio {key} {fname} failed...", flush=True)
        if samps is None:
            raise RuntimeError("Audio IO failed ...")
        if self.ch >= 0 and samps.ndim == 2:
            samps = samps[self.ch]
        return samps

    def nsamps(self, key: str) -> int:
        """
        Number of samples
        """
        data = self._load(key)
        return data.shape[-1]

    def power(self, key: str) -> float:
        """
        Power of utterance
        """
        data = self._load(key)
        s = data if data.ndim == 1 else data[0]
        return np.linalg.norm(s, 2)**2 / data.size

    def duration(self, key: str) -> float:
        """
        Utterance duration
        """
        N = self.nsamps(key)
        return N / self.sr
