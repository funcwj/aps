#!/usr/bin/env python

# wujian@2019
"""
Dataloader for online simulation (slow, need to be optimized)
"""
import json
import glob

import numpy as np
import scipy.signal as ss

import torch as th
import torch.utils.data as dat

from .wave import read_wav, DataLoader, EPSILON
from .utils import BatchSampler

from kaldi_python_io import Reader as BaseReader
"""
Configuration egs: (Generate by setk/scripts/sptk/create_data_conf.py)
{
"key": "f36d634d-b5f3-492b-888b-0d30269f39e4",
"len": 96350,
"nch": 4,
"dur": 6.022,
"scl": 0.65,
"spk": [
    {
    "doa": 117.927,
    "rir": "/home/work_nfs3/jwu/workspace/far_simu/gpu_rir/Room194-28.wav",
    "loc": "/home/work_nfs/common/data/AIShell-2-Eval-Test/TEST/IOS/T0023/IT0023W0269.wav",
    "txt": "3619 2028 4172 897 4608 136 608 1542 153 3504 2123 1598 1615 3079 58 2978 462",
    "len": 96350,
    "sdr": 0,
    "off": 0
    }
],
"iso": {
    "snr": 17.789,
    "beg": 349312,
    "end": 445662,
    "loc": "/home/work_nfs3/jwu/workspace/far_simu/iso/iso3.wav",
    "off": 0
},
"ptn": {
    "snr": 7.71,
    "beg": 41229,
    "end": 137579,
    "loc": "/home/backup_nfs/data-ASR/NoiseAndRIRs/Musan/noise/noise/free-sound/noise-free-sound-0250.wav",
    "off": 0,
    "rir": "/home/work_nfs3/jwu/workspace/far_simu/gpu_rir/Room194-19.wav",
    "doa": 273.506
}
}
"""


def conf_loader(simu_conf="",
                train=True,
                channel=-1,
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
                                channel=channel,
                                sr=sr,
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
    # revb = [ss.fftconvolve(spk, r)[:S] for r in rir]
    revb = ss.oaconvolve(spk[None, ...], rir)[..., :S]
    revb = np.asarray(revb)

    if early_energy:
        rir_ch0 = rir[0]
        rir_peak = np.argmax(rir_ch0)
        rir_beg_idx = max(0, int(rir_peak - 0.001 * sr))
        rir_end_idx = min(rir_ch0.size, int(rir_peak + 0.05 * sr))
        early_rir = rir_ch0[rir_beg_idx:rir_end_idx]
        early_rev = ss.oaconvolve(spk, early_rir)[:S]
        return revb, np.mean(early_rev**2)
    else:
        return revb, np.mean(revb[0]**2)

def snr_coeff(sig_pow, ref_pow, snr):
    """
    Compute signal scale factor according to snr
    """
    return (ref_pow / (sig_pow * 10**(snr / 10) + EPSILON))**0.5


def process_conf(simu_conf,
                 max_token_num=400,
                 min_token_num=2,
                 max_dur=30,
                 min_dur=0.4):
    conf_set = []
    for idx, conf in enumerate(simu_conf):
        # speaker token
        token_str = conf["spk"][0]["txt"]
        token = [int(n) for n in token_str.split()]
        num_tokens = len(token)
        if num_tokens > max_token_num or num_tokens <= min_token_num:
            continue
        # mixture duration
        cur_dur = conf["dur"]
        if cur_dur < min_dur or cur_dur > max_dur:
            continue
        conf_set.append({
            "idx": idx,
            "dur": cur_dur,
            "tok": token,
            "len": num_tokens
        })
    # long -> short
    conf_set = sorted(conf_set, key=lambda d: d["dur"], reverse=True)
    if len(conf_set) < 10:
        raise RuntimeError("Too less utterances, check data configurations")
    return conf_set


class SimulationDataset(dat.Dataset):
    """
    A specific dataset performing online data simulation, configured by a json file
    """
    def __init__(self,
                 simu_conf,
                 channel=-1,
                 sr=16000,
                 max_token_num=400,
                 max_wav_dur=30,
                 min_wav_dur=0.4):
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
        self.sr = sr
        self.ch = channel
        self.step()

    def step(self):
        cur_conf = self.epoch_conf[self.epoch]
        with open(cur_conf, "r") as f:
            self.cur_simu_conf = json.load(f)
            self.token_reader = process_conf(self.cur_simu_conf, **self.proc_kwargs)
            self.epoch = (self.epoch + 1) % len(self.epoch_conf)
        print("Got dataset from configuration: " + 
                f"{cur_conf}, {len(self.token_reader)} utterances", flush=True)

    def __len__(self):
        # make sure that each conf has same utterances
        return len(self.token_reader)

    def _simu(self, conf):
        mix_len = conf["len"]
        spk_image = []
        spk_power = []
        spk_meta = conf["spk"]

        for sconf in spk_meta:
            rir = None
            spk = read_wav(sconf["loc"])
            # if has rir
            if "rir" in sconf:
                rir = read_wav(sconf["rir"])
                if self.ch >= 0:
                    if rir.ndim == 2:
                        rir = rir[self.ch:self.ch + 1]
                    else:
                        rir = rir[None, ...]
                revb, p = add_room_response(spk, rir, sr=self.sr)
                spk_image.append(revb)
                spk_power.append(p)
            else:
                spk_image.append(spk)
                spk_power.append(np.mean(spk**2))

        # reference power, use ch0
        ref_pow = spk_power[0]
        ref = []
        pad_shape = (conf["nch"] if not self.ch >= 0 else 1, mix_len)
        # scale spk{1..N} by sdr
        for i, s in enumerate(spk_meta):
            spk_pad = np.zeros(pad_shape, dtype=np.float32)
            spk_len = min(spk_image[i].shape[-1], mix_len)
            if i == 0:
                spk_pad[:, :spk_len] = spk_image[i][:, :spk_len]
            else:
                cur_pow = spk_power[i]
                coeff = snr_coeff(cur_pow, ref_pow, s["sdr"])
                beg = s["off"]
                spk_pad[:, beg:beg +
                        spk_len] = coeff * spk_image[i][:, :spk_len]
            ref.append(spk_pad)
        # mixed
        mix = sum(ref)
        # add noise
        for noise_type in ["iso", "ptn"]:
            if noise_type in conf:
                cfg = conf[noise_type]
                # read a segment (make IO efficient)
                ptn = read_wav(cfg["loc"], beg=cfg["beg"], end=cfg["end"])
                if noise_type == "ptn":
                    # convolve rir
                    if "rir" in cfg:
                        rir = read_wav(cfg["rir"])
                        if self.ch >= 0:
                            if rir.ndim == 2:
                                rir = rir[self.ch:self.ch + 1]
                            else:
                                rir = rir[None, ...]
                        ptn, ptn_pow = add_room_response(ptn, rir, sr=self.sr)
                    else:
                        ptn_pow = np.mean(ptn**2)
                        ptn = ptn[None, ...]
                else:
                    ptn_pow = np.mean(ptn[0]**2)
                if self.ch >= 0:
                    ptn = ptn[0:1]
                coeff = snr_coeff(ptn_pow, ref_pow, cfg["snr"])
                ptn_len = ptn.shape[-1]
                ptn_off = cfg["off"]
                mix[:, ptn_off:ptn_off + ptn_len] += coeff * ptn

        factor = conf["scl"] / max(np.max(np.abs(mix[0])), EPSILON)
        if self.ch >= 0:
            return mix[0] * factor
        else:
            return mix * factor

    def __getitem__(self, idx):
        token = self.token_reader[idx]
        conf = self.cur_simu_conf[token["idx"]]
        wav = self._simu(conf)
        return {
            "dur": wav.shape[-1],
            "len": token["len"],
            "wav": wav,
            "tok": token["tok"]
        }
