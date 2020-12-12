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


@ApsRegisters.loader.register("ss_online")
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
    """
    dataset = SimuOptsDataset(simu_cfg, noise=noise_label)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


class SimuOptsDataset(dat.Dataset):
    """
    Dataset configured by the simulation command options
    """

    def __init__(self, simu_cfg, noise: bool = False) -> None:
        self.simu_cfg = BaseReader(simu_cfg, num_tokens=-1)
        self.noise = noise
        self.parser = make_argparse()

    def _simu(self, opts_str: str) -> Dict:
        args = self.parser.parse_args(opts_str)
        mix, spk_ref, noise = run_simu(args)
        if self.noise and noise is not None:
            spk_ref.append(noise)
        if len(spk_ref) == 1:
            spk_ref = spk_ref[0]
        egs = {"mix": mix, "ref": spk_ref}
        return egs

    def __getitem__(self, index: int) -> Dict:
        return self._simu(self.simu_cfg[index])

    def __len__(self) -> int:
        return len(self.simu_cfg)

    def __iter__(self) -> Iterator[Dict]:
        for _, opts_str in self.simu_cfg:
            yield self._simu(opts_str)
