#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Online simulation dataloader for speech enhancement & separation tasks
"""

import torch.utils.data as dat

from typing import Dict, Iterable, List, Iterator
from kaldi_python_io import Reader as BaseReader

from aps.loader.se.chunk import WaveChunkDataLoader
from aps.libs import ApsRegisters
from aps.loader.simu import run_simu, make_argparse


@ApsRegisters.loader.register("se@online")
def DataLoader(train: bool = True,
               sr: int = 16000,
               simu_cfg: str = "",
               noise_label: bool = False,
               chunk_size: int = 64000,
               batch_size: int = 16,
               distributed: bool = False,
               num_workers: int = 4) -> Iterable[Dict]:
    """
    Return a online simulation dataloader for enhancement/separation tasks
    Args
        train: in training mode or not
        sr: sample rate of the audio
        simu_cfg: configuration file for online data simulation
        chunk_size: #chunk_size (s)
        batch_size: #batch_size
        distributed: in distributed mode or not
        num_workers: number of workers used in dataloader
    """
    dataset = SimuOptionsDataset(simu_cfg,
                                 return_in_egs=["mix", "ref", "noise"]
                                 if noise_label else ["mix", "ref"])
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


class SimuOptionsDataset(dat.Dataset):
    """
    A dataset drived by the simulation configurations. The format of the "simu.cfg" looks like
        2spk_mix02  --src-spk /path/to/XXXX1.wav,/path/to/XXXX2.wav --src-begin 4000,0 --src-sdr 1 ...
        2spk_mix03  --src-spk /path/to/XXXX3.wav,/path/to/XXXX4.wav --src-begin 0,900 --src-sdr -1 ...
        ...
    or
        1spk_simu1  --src-spk /path/to/XXXX1.wav --src-rir /path/to/rir1.wav --point-noise /path/to/noise1.wav \
                    --point-noise-snr 5.1 --point-noise-rir /path/to/rir2.wav
        1spk_simu2  --src-spk /path/to/XXXX2.wav --src-rir /path/to/rir3.wav --point-noise /path/to/noise2.wav \
                    --point-noise-snr 2.4 --point-noise-rir /path/to/rir4.wav
    where each line follows pattern <key> <command-options>. Please refer details in
    1) aps/loader/simu.py
    2) https://github.com/funcwj/setk/tree/master/doc/data_simu for command line usage

    The user should write the script to generate the simu.cfg based on the requirements.
    Args:
        simu_cfg: path of the audio simulation configuraton file
        return_in_egs: mix|ref|noise, return mixture, reference, noise signals or not
    """

    def __init__(self,
                 simu_cfg: str,
                 return_in_egs: List[str] = ["mix"]) -> None:
        self.simu_cfg = BaseReader(simu_cfg, num_tokens=-1)
        self.parser = make_argparse()
        self.return_in_egs = return_in_egs

    def _simu(self, opts_str: str) -> Dict:
        """
        Args:
            opts_str: command options for aps/loader/simu.py
        Return:
            egs: training egs
        """
        args = self.parser.parse_args(opts_str)
        mix, spk_ref, noise = run_simu(args)
        egs = {"mix": mix}
        if "noise" in self.return_in_egs and noise is not None:
            spk_ref.append(noise)
        if "ref" in self.return_in_egs:
            egs["ref"] = spk_ref[0] if len(spk_ref) == 1 else spk_ref
        return egs

    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index: index ID
        Return:
            egs: training egs
        """
        opts_str = self.simu_cfg[index]
        return self._simu(opts_str)

    def __len__(self) -> int:
        """
        Return number of utterances
        """
        return len(self.simu_cfg)

    def __iter__(self) -> Iterator[Dict]:
        """
        Return audio chunk iterator
        """
        for _, opts_str in self.simu_cfg:
            yield self._simu(opts_str)
