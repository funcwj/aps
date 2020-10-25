#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Online simulation dataloader for speech enhancement & separation tasks
"""

from kaldi_python_io import Reader as BaseReader

from aps.loader.simu import run_simu, make_argparse
from aps.loader.ss.chunk import type_seq, WaveChunkDataLoader


def DataLoader(train=True,
               sr=16000,
               simu_cfg="",
               noise_label=False,
               chunk_size=64000,
               batch_size=16,
               distributed=False,
               num_workers=4):
    """
    Return a online simulation dataloader for enhancement/separation tasks
    """
    dataset = SimuOptsDataset(simu_cfg=simu_cfg, noise=noise_label)
    return WaveChunkDataLoader(dataset,
                               train=train,
                               chunk_size=chunk_size,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               distributed=distributed)


class SimuOptsDataset(object):
    """
    Dataset configured by the simulation command options
    """

    def __init__(self, simu_cfg="", noise=False):
        self.simu_cfg = BaseReader(simu_cfg, num_tokens=-1)
        self.noise = noise
        self.parser = make_argparse()

    def _simu(self, opts_str):
        args = self.parser.parse_args(opts_str)
        mix, spk_ref, noise = run_simu(args)
        if self.noise and noise is not None:
            spk_ref.append(noise)
        if len(spk_ref) == 1:
            spk_ref = spk_ref[0]
        egs = {"mix": mix, "ref": spk_ref}
        return egs

    def __getitem__(self, index):
        return self._simu(self.simu_cfg[index])

    def __len__(self):
        return len(self.simu_cfg)

    def __iter__(self):
        for _, opts_str in self.simu_cfg:
            yield self._simu(opts_str)
