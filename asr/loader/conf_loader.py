#!/usr/bin/env python

# wujian@2019
"""
Dataloader for online simulation
"""
import json
import glob

import numpy as np
import scipy.signal as ss

import torch as th
import torch.utils.data as dat

from .wav_loader import read_wav, DataLoader, EPSILON
from .utils import BatchSampler

from kaldi_python_io import Reader as BaseReader
"""
simulation configuration looks like
{
"key": "000ce6fa-2fac-4976-b3a0-8e252fd8517c",
"len": 85632,   # number of samples
"dur": 7.025,   # duration
"scl": 0.64,    # scale
"spk": {
    "doa": 165.935,
    "key": "XXXXXXXXXXX",
    "rir": /path/to/rir/spkXXXXX.wav",
    "loc": /path/to/spk/spkXXXXX.wav",
}
"dir": {
    "doa": 42.231,
    "rir": "/path/to/rir/dirXXXX.wav",
    "loc": "/path/to/dir/dirXXXX.wav",
    "len": 85632,   # length of the noise
    "snr": 3.672,
    "off": 0        # offset to add to the source speaker
}
"iso": {
    "snr": 17.316,
    "beg": 186993,
    "loc": "/path/to/dir/iso_noise_XXXX.wav"
}
}
"""


def make_online_loader(simu_conf="",
                       token="",
                       train=True,
                       single_channel=False,
                       add_rir=True,
                       sr=16000,
                       max_token_num=400,
                       max_dur=30,
                       min_dur=0.4,
                       adapt_dur=8,
                       adapt_token_num=150,
                       batch_size=32,
                       num_workers=4,
                       min_batch_size=4):
    dataset = SimulationDataset(simu_conf,
                                token,
                                single_channel=single_channel,
                                sr=sr,
                                add_rir=add_rir,
                                max_token_num=max_token_num,
                                max_wav_dur=max_dur,
                                min_wav_dur=min_dur)
    return DataLoader(dataset,
                      shuffle=train,
                      num_workers=num_workers,
                      adapt_wav_dur=adapt_dur,
                      adapt_token_num=adapt_token_num,
                      batch_size=batch_size,
                      min_batch_size=min_batch_size)


def add_room_response(spk, rir, early_energy=True, sr=16000):
    """
    Convolute source signal with selected rirs
    args
        spk: S
        rir: N x R
    return
        revb: N x S
    """
    S = spk.shape[-1]
    revb = [ss.fftconvolve(spk, r)[:S] for r in rir]
    revb = np.asarray(revb)

    if early_energy:
        rir_ch0 = rir[0]
        rir_peak = np.argmax(rir_ch0)
        rir_beg_idx = max(0, int(rir_peak - 0.001 * sr))
        rir_end_idx = min(rir_ch0.size, int(rir_peak + 0.05 * sr))
        early_rir = rir_ch0[rir_beg_idx:rir_end_idx]
        early_rev = ss.fftconvolve(spk, early_rir)[:S]
        return revb, np.mean(early_rev**2)
    else:
        return revb, np.mean(revb[0]**2)


def process_token(token_reader,
                  simu_conf,
                  max_token_num=400,
                  min_token_num=2,
                  max_dur=30,
                  min_dur=0.4):
    token_set = []
    for idx, conf in enumerate(simu_conf):
        tok_key = conf["spk"]["key"]
        token = token_reader[tok_key]
        num_tokens = len(token)
        if num_tokens > max_token_num or num_tokens <= min_token_num:
            continue
        cur_dur = conf["len"]
        if cur_dur < min_dur or cur_dur > max_dur:
            continue
        token_set.append({
            "key": idx,
            "dur": cur_dur,
            "tok": token,
            "len": num_tokens
        })
    # long -> short
    token_set = sorted(token_set, key=lambda d: d["dur"], reverse=True)
    if len(token_set) < 10:
        raise RuntimeError("Too less utterances, check data configurations")
    return token_set


class SimulationDataset(dat.Dataset):
    """
    A specific dataset performing the online data simulation, configured by a yaml file
    """
    def __init__(self,
                 simu_conf,
                 token,
                 single_channel=False,
                 sr=16000,
                 add_rir=True,
                 max_token_num=400,
                 max_wav_dur=30,
                 min_wav_dur=0.4):
        self.token_reader = BaseReader(
            token,
            value_processor=lambda l: [int(n) for n in l],
            num_tokens=-1,
            restrict=False)
        # fetch matched configurations
        self.epoch_conf = sorted(glob.glob(simu_conf))
        self.epoch = 0
        if len(self.epoch_conf) == 0:
            raise RuntimeError(
                f"Can't find matched configuration pattern: {simu_conf}")
        self.proc_kwargs = {
            "max_token_num": max_token_num,
            "min_token_num": 2,
            "max_dur": max_wav_dur,
            "min_dur": min_wav_dur
        }
        with open(self.epoch_conf[0], "r") as f:
            self.cur_simu_conf = json.load(f)
        self.token_set = process_token(self.token_reader, self.cur_simu_conf,
                                       **self.proc_kwargs)
        self.sr = sr
        # Add rir or not (must single channel)
        self.add_rir = add_rir
        # keep multi-channel signal or not
        self.single_channel = single_channel

    def step(self):
        cur_conf = self.epoch_conf[self.epoch]
        with open(cur_conf, "r") as f:
            self.cur_simu_conf = json.load(f)
        print(f"Got dataset from configuration: {cur_conf}", flush=True)
        self.token_set = process_token(self.token_reader, self.cur_simu_conf,
                                       **self.proc_kwargs)
        # step to next one
        self.epoch = (self.epoch + 1) % len(self.epoch_conf)

    def __len__(self):
        # make sure that each conf has same utterances
        return len(self.token_set)

    def _load(self, conf):
        """
        Read wav data, partially or fully
        """
        if "beg" in conf:
            # patial
            if "len" not in conf:
                raise RuntimeError(f"Missing key => \"len\" in conf: {conf}")
            beg = conf["beg"]
            wav = read_wav(conf["loc"], beg=beg, end=beg + conf["len"])
        else:
            wav = read_wav(conf["loc"])
        return wav

    def _conv(self, conf):
        """
        Return convolved signals
        """
        src = self._load(conf["loc"])
        # if use rir
        if self.add_rir:
            rir = self._load(conf["rir"])
            # single-channel, just use ch0
            if self.single_channel:
                if rir.ndim == 2:
                    rir = rir[0:1]
                else:
                    rir = rir[None, ...]
            # make sure rir in N x R
            src_reverb, src_pow = add_room_response(src, rir, sr=self.sr)
        else:
            src_reverb = src
            src_pow = np.mean(src**2)
        return src_reverb, src_pow

    def _simu(self, conf):
        # convolved (or not) speaker
        spk_reverb, spk_pow = self._conv(conf["spk"])
        # convolved (or not) noise
        if "dir" in cur_dir:
            cur_dir = conf["dir"]
            dir_reverb, dir_pow = self._conv(cur_dir)
            # add noise
            dir_snr = cur_dir["snr"]
            dir_beg = cur_dir["off"]
            dir_scl = (spk_pow / (dir_pow * 10**(dir_snr / 10) + EPSILON))**0.5
            spk_reverb[:, dir_beg:dir_beg +
                       cur_dir["len"]] += dir_scl * dir_reverb
        # add isotropic noise if needed
        if "iso" in conf:
            cur_iso = conf["iso"]
            mix_len = conf["len"]
            iso = self._load(cur_iso)
            if self.single_channel:
                iso = iso[0:1]
            # prepare iso noise
            pad_size = mix_len - iso.shape[-1]
            if pad_size > 0:
                pad_width = ((0, 0), (0, pad_size))
                iso_pad = np.pad(iso, pad_width, mode="wrap")
            else:
                iso_pad = iso[:, :mix_len]
            # add noise
            iso_pow = np.mean(iso_pad[0]**2)
            iso_scl = (spk_pow /
                       (iso_pow * 10**(cur_iso["snr"] / 10) + EPSILON))**0.5
            spk_reverb += iso_scl * iso_pad
        spk_scale = conf["scl"] / (np.max(np.abs(spk_reverb[0])) + EPSILON)
        spk_reverb = spk_reverb * spk_scale
        if self.single_channel:
            return spk_reverb[0]
        else:
            return spk_reverb

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        key = tok["key"]
        cur_conf = self.cur_simu_conf[key]
        wav = self._simu(cur_conf)
        return {
            "dur": wav.shape[-1],
            "len": tok["len"],
            "wav": wav,
            "tok": tok["tok"]
        }
