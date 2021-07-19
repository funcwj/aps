#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import json
import gzip
import h5py

import numpy as np
import torch.utils.data as dat

from aps.loader.audio import add_room_response
from aps.loader.simu import coeff_snr
from aps.loader.se.chunk import WaveChunkDataLoader
from aps.const import MAX_INT16, EPSILON
from aps.libs import ApsRegisters

from typing import Dict, List, Tuple, Iterable


@ApsRegisters.loader.register("se@config")
def DataLoader(train: bool = True,
               simu_cfg: str = "",
               single_channel: bool = False,
               max_num_speakers: int = 2,
               hdf5_key: str = "wav",
               sr: int = 16000,
               early_reverb: bool = False,
               noise_reference: bool = True,
               rir_prob: float = 1.0,
               isotropic_noise_prob: float = 1.0,
               directional_noise_prob: float = 1.0,
               chunk_size: int = 64000,
               max_batch_size: int = 16,
               distributed: bool = False,
               num_workers: int = 4) -> Iterable[Dict]:
    """
    Return a online simulation dataloader for enhancement/separation tasks (json version)
    Args
        train: in training mode or not
        simu_cfg: configuration file for online data simulation
        single_channel: force single channel audio if multi-channel rir is given
        max_num_speakers: max number of speakers created
        sr: sample rate of the audio
        early_revb_target: using early reverberated signal as training target
        rir_prob: probability of adding rirs
        isotropic_noise_prob: probability of adding isotropic noises
        directional_noise_prob: probability of adding directional noises
        chunk_size: #chunk_size (s)
        max_batch_size: #batch_size
        distributed: in distributed mode or not
        num_workers: number of workers used in dataloader
    """
    dataset = ConfigSimulationDataset(
        simu_cfg,
        single_channel=single_channel,
        max_num_speakers=max_num_speakers,
        hdf5_key=hdf5_key,
        sr=sr,
        early_reverb=early_reverb,
        noise_reference=noise_reference,
        rir_prob=rir_prob if train else 0,
        isotropic_noise_prob=isotropic_noise_prob if train else 1,
        directional_noise_prob=directional_noise_prob if train else 1)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=max_batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


"""
Json configuration examples
[
  {
    "key": "c762c7fd-8dbd-8ff2-8c38-2cb9cfb54140",
    "length": 112320,
    "inf_norm": 0.5247265505241899,
    "num_speakers": 2,
    "rir_channels": 4,
    "speakers": [
      {
        "doa": 142.9514916241633,
        "utt": "/path/to/wav/hdf5:key:206913120:207025440",
        "rir": "/path/to/rir/hdf5:key:1734350:1743337",
        "sdr": 0.0,
        "offset": 0
      },
      {
        "doa": 135,
        "utt": "/path/to/wav/hdf5:key:157627680:157695360",
        "rir": "/path/to/rir/hdf5:key:1851181:1860168",
        "sdr": 4.091795928395875,
        "offset": 0
      }
    ],
    "directional_noise": [
      {
        "utt": "/path/to/directional_noise/hdf5:key:19098288:19258288",
        "rir": "/path/to/rir/hdf5:key:1968012:1976999",
        "snr": 9.783469850151286,
        "truncated": "10640:35406,44101:85483",
        "offset": "20968,107122"
      },
      {
        "utt": "/path/to/directional_noise/hdf5:key:5919999:6079999",
        "rir": "/path/to/rir/hdf5:key:2237622:2246609",
        "snr": 5.328025441060883,
        "truncated": "18708:43784,15652:54327",
        "offset": "53272,41584"
      },
      {
        "utt": "/path/to/directional_noise/hdf5:key:166584899:166744899",
        "rir": "/path/to/rir/hdf5:key:2264583:2273570",
        "snr": 2.5913184936895584,
        "truncated": "82607:107474",
        "offset": "15693"
      }
    ],
    "isotropic_noise": {
      "snr": 15.838126188253053,
      "utt": "/path/to/isotropic_noise/hdf5:key:320000:640000",
      "truncated": 130842
    }
  },
  ...
]
"""


class ConfigSimulationDataset(dat.Dataset):
    """
    Online data simulation dataset with json configuration

    Args:
        simu_cfg: Simulation configuration file
        single_channel: force single channel audio if multi-channel rir is given
        max_num_speakers: max number of speakers created
        sr: sample rate of the audio
        early_revb_target: using early reverberated signal as training target
        rir_prob: probability of adding rirs
        isotropic_noise_prob: probability of adding isotropic noises
        directional_noise_prob: probability of adding directional noises
    """

    def __init__(self,
                 simu_cfg: str,
                 single_channel: bool = False,
                 max_num_speakers: int = 2,
                 hdf5_key: str = "wav",
                 sr: int = 16000,
                 early_reverb: bool = False,
                 noise_reference: bool = True,
                 rir_prob: float = 1.0,
                 isotropic_noise_prob: float = 1.0,
                 directional_noise_prob: float = 1.0):
        self.simu_cfg = self._load_cfg(simu_cfg)
        self.sr = sr
        self.key = hdf5_key
        self.container = {}
        self.force_single = single_channel
        self.early_reverb = early_reverb
        self.max_spks = max_num_speakers
        self.rir_prob = rir_prob
        self.iso_noise_prob = isotropic_noise_prob
        self.dir_noise_prob = directional_noise_prob
        self.noise_ref = noise_reference

    def _load_cfg(self, simu_cfg: str) -> List:
        """
        Load json from .json or .json.gz (gzip compressed)
        """
        if simu_cfg[-2:] == "gz":
            with gzip.open(simu_cfg, "r") as fp:
                raw = fp.read()
                cfg = json.loads(raw)
        else:
            with open(simu_cfg, "r") as fp:
                cfg = json.load(fp)
        return cfg

    def _load_audio(self,
                    cfg: str,
                    dtype: str,
                    offset: int = 0,
                    length: int = -1) -> np.ndarray:
        """
        Load rir (C x S) from current rir_cfg
        """
        assert dtype in ["rir", "spk", "dir", "iso"]
        ark_addr, _, beg, end = cfg.split(":")
        beg, end = int(beg), int(end)
        # cache
        if ark_addr not in self.container:
            self.container[ark_addr] = h5py.File(ark_addr, "r")[self.key]
        utt_chunk = self.container[ark_addr]
        # add offset
        beg += offset
        if length > 0:
            end = min(end, beg + length)
        audio = utt_chunk[..., beg:end]
        if self.force_single and dtype in ["rir", "iso"]:
            if audio.ndim == 2:
                audio = audio[0:1]
            else:
                audio = audio[None, ...]
        audio = audio.astype(np.float32) / MAX_INT16
        return audio

    def _conv_speaker_with_rir(self, cfg: str, add_rir: bool = True):
        """
        Convolve source speaker with rir if needed
        """
        # load source speakers
        spk = self._load_audio(cfg["utt"], "spk")
        if add_rir and "rir" in cfg:
            # load rir
            rir = self._load_audio(cfg["rir"], "rir")
            reverb, early_reverb, power = add_room_response(
                spk,
                rir,
                early_energy=self.early_reverb,
                sr=self.sr,
                early_revb_duration=0.05)
            if self.early_reverb:
                return (reverb, early_reverb, power)
            else:
                return (reverb, reverb[0], power)
        else:
            return (spk, spk, np.mean(spk**2))

    def _conv_zero_with_rir(self, shape: Tuple, add_rir: bool = True):
        """
        Convolve zero signal with rir if needed
        """
        early_reverb = np.zeros(shape[-1], dtype=np.float32)
        if add_rir and not self.force_single:
            reverb = np.zeros(shape, dtype=np.float32)
        else:
            reverb = np.zeros_like(early_reverb)
        return reverb, early_reverb, 0

    def _mix_speakers(self, spk_stats: List, cfg: List, shape: Tuple,
                      ref_power: float):
        """
        Mix each speakers and kept reference
        """
        ref_revb = []
        ref_early_revb = []
        spk_lens = [s[1].shape[-1] for s in spk_stats]
        num_spks = len(spk_stats)
        for i, cur_cfg in enumerate(cfg):
            cur_len = spk_lens[i]
            spk_reverb, spk_early_reverb, cur_power = spk_stats[i]
            spk_pad = np.zeros(shape, dtype=np.float32)
            spk_early_revb_pad = np.zeros(shape[-1], dtype=np.float32)
            if i == 0:
                spk_pad[:, :cur_len] = spk_reverb
                spk_early_revb_pad[:cur_len] = spk_early_reverb
            else:
                scale = coeff_snr(cur_power, ref_power, cur_cfg["sdr"])
                beg = cur_cfg["offset"]
                spk_pad[:, beg:beg + cur_len] = scale * spk_reverb
                spk_early_revb_pad[beg:beg + cur_len] = scale * spk_early_reverb
            ref_revb.append(spk_pad)
            ref_early_revb.append(spk_early_revb_pad)
        for i in range(len(cfg), num_spks):
            ref_revb.append(spk_stats[i][0])
            ref_early_revb.append(spk_stats[i][1])
        return sum(ref_revb), ref_early_revb

    def _load_isotropic_noise(self, cfg: Dict, shape: Tuple, ref_power: float):
        """
        Load isotropic noise if needed
        """
        # 1 x S or C x S
        ref_iso_noise = np.zeros(shape, dtype=np.float32)
        add_iso_noise = np.random.binomial(1, self.iso_noise_prob)
        if "isotropic_noise" in cfg and add_iso_noise:
            cfg = cfg["isotropic_noise"]
            mix_len = shape[-1]
            iso = self._load_audio(cfg["utt"],
                                   "iso",
                                   offset=cfg["truncated"],
                                   length=mix_len)
            iso_len = iso.shape[-1]
            pad_size = mix_len - iso_len
            if pad_size > 0:
                pad_width = ((0, 0), (0, pad_size))
                iso_pad = np.pad(iso, pad_width, mode="wrap")
            else:
                iso_pad = iso[:, :mix_len]
            iso_power = np.mean(iso_pad[0]**2)
            scale = coeff_snr(iso_power, ref_power, cfg["snr"])
            ref_iso_noise += scale * iso_pad
        return ref_iso_noise

    def _load_directional_noise(self,
                                cfg: Dict,
                                shape: Tuple,
                                ref_power: float,
                                add_rir: bool = True) -> np.ndarray:
        """
        Load directional noise if needed
        """
        # 1 x S or C x S
        ref_dir_noise = np.zeros(shape, dtype=np.float32)
        add_dir_noise = np.random.binomial(1, self.dir_noise_prob)
        if "directional_noise" in cfg and add_dir_noise:
            cfg = cfg["directional_noise"]
            for dir_cfg in cfg:
                # load stats
                tokens = dir_cfg["truncated"].split(",")
                seg_beg_end = [tuple(map(int, t.split(":"))) for t in tokens]
                seg_len = [end - beg for beg, end in seg_beg_end]
                mix_beg = tuple((map(int, dir_cfg["offset"].split(","))))
                mix_end = [b + seg_len[i] for i, b in enumerate(mix_beg)]
                for i in range(len(seg_len)):
                    seg_beg, _ = seg_beg_end[i]
                    # load noise chunk
                    cut_noise = self._load_audio(dir_cfg["utt"],
                                                 "dir",
                                                 offset=seg_beg,
                                                 length=seg_len[i])
                    # add rir or not
                    if add_rir and "rir" in dir_cfg:
                        rir = self._load_audio(cfg["rir"], "rir")
                        revb_noise, _, noise_power = add_room_response(
                            cut_noise,
                            rir,
                            early_energy=False,
                            early_revb_target=False,
                            sr=self.sr)
                    else:
                        revb_noise = cut_noise[None, ...]
                        noise_power = np.mean(cut_noise**2)
                    scale = coeff_snr(noise_power, ref_power, dir_cfg["snr"])
                    ref_dir_noise[:,
                                  mix_beg[i]:mix_end[i]] += scale * revb_noise
        return ref_dir_noise

    def _prepare_egs(self,
                     mix: np.ndarray,
                     ref: List[np.ndarray],
                     dir_noise: np.ndarray,
                     iso_noise: np.ndarray,
                     inf_norm: float = 0.8):
        """
        Prepare audio egs
        """
        mix = mix + dir_noise + iso_noise
        scale = 1 if inf_norm == 0 else inf_norm / np.max(
            np.abs(mix[0]) + EPSILON)
        if self.noise_ref:
            ref.append(dir_noise[0] + iso_noise[0])
        if self.force_single:
            mix = mix[0]
        ref = [r * scale for r in ref]
        if len(ref) == 1:
            ref = ref[0]
        egs = {"mix": mix * scale, "ref": ref}
        return egs

    def _simu(self, cfg: Dict) -> Dict:
        """
        Audio simulation
        """
        utt_shape = (cfg["num_channels"] if not self.force_single else 1,
                     cfg["length"])
        # add rir or not
        add_rir = np.random.binomial(1, self.rir_prob)
        # cache (reverb, early_reverb, power)
        spk_stats = []
        # conv speakers
        for spk_cfg in cfg["speakers"]:
            spk_stats.append(
                self._conv_speaker_with_rir(spk_cfg, add_rir=add_rir))
        # conv zero signals
        for _ in range(self.max_spks - cfg["num_speakers"]):
            spk_stats.append(
                self._conv_zero_with_rir(utt_shape, add_rir=add_rir))
        # mix spk{1..N} by sdr
        ref_power = spk_stats[0][-1]
        mix, ref = self._mix_speakers(spk_stats, cfg["speakers"], utt_shape,
                                      ref_power)
        # load isotropic noise
        iso_noise = self._load_isotropic_noise(cfg, utt_shape, ref_power)
        # load directional noise
        dir_noise = self._load_directional_noise(cfg, utt_shape, ref_power)
        # generate egs
        egs = self._prepare_egs(mix,
                                ref,
                                dir_noise,
                                iso_noise,
                                inf_norm=cfg["inf_norm"])
        # add utterance key
        egs["key"] = cfg["key"]
        return egs

    def __len__(self) -> int:
        return len(self.simu_cfg)

    def __getitem__(self, index):
        return self._simu(self.simu_cfg[index])
