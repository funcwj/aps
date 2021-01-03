#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Online simulation dataloader for speech enhancement & separation tasks
"""

import torch.utils.data as dat

from kaldi_python_io import Reader as BaseReader
from typing import Dict, Iterator, Iterable
from aps.loader.simu import run_simu, make_argparse
from aps.loader.se.chunk import WaveChunkDataLoader
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("se_online")
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
    dataset = SimuOptionsDataset(simu_cfg, noise=noise_label)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


class SimuOptionsDataset(dat.Dataset):
    """
    A dataset drived by the simulation configurations. The format of the "simu.cfg" looks like
        2spk_mix02  --src-spk /path/to/XXXX1.wav,/path/to/XXXX2.wav --src-begin 4000,0 --src-sdr 1
        2spk_mix03  --src-spk /path/to/XXXX3.wav,/path/to/XXXX4.wav --src-begin 0,900 --src-sdr -1
        ...
    where each line follows pattern <key> <command-options>. See
    1) aps/loader/simu.py for <command-options> details
    2) https://github.com/funcwj/setk/tree/master/doc/data_simu for command line usage

    Args:
        simu_cfg: path of the audio simulation configuraton file
        noise: if true, then return both noise and reference audio
    """

    def __init__(self, simu_cfg: str, noise: bool = False) -> None:
        self.simu_cfg = BaseReader(simu_cfg, num_tokens=-1)
        self.noise = noise
        self.parser = make_argparse()

    def _simu(self, opts_str: str) -> Dict:
        """
        Args:
            opts_str: command options for aps/loader/simu.py
        Return:
            egs: training egs
        """
        args = self.parser.parse_args(opts_str)
        mix, spk_ref, noise = run_simu(args)
        if self.noise and noise is not None:
            spk_ref.append(noise)
        if len(spk_ref) == 1:
            spk_ref = spk_ref[0]
        egs = {"mix": mix, "ref": spk_ref}
        return egs

    def __getitem__(self, index: int) -> Dict:
        """
        Args:
            index: index ID
        """
        return self._simu(self.simu_cfg[index])

    def __len__(self) -> int:
        return len(self.simu_cfg)

    def __iter__(self) -> Iterator[Dict]:
        """
        Return audio chunk iterator
        """
        for _, opts_str in self.simu_cfg:
            yield self._simu(opts_str)
