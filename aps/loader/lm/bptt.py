#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
for RNNLM (BPTT training)
"""

import torch.utils.data as dat

import aps.distributed as dist

from typing import Optional, Iterable, Iterator, Dict, List, NoReturn
from aps.loader.lm.utils import filter_utts, concat_data
from aps.loader.am.utils import derive_indices
from aps.loader.lm.utt import Dataset
from aps.utils import get_logger
from aps.libs import ApsRegisters

logger = get_logger(__name__)


@ApsRegisters.loader.register("lm@bptt")
def DataLoader(text: str = "",
               vocab_dict: Optional[Dict] = None,
               train: bool = True,
               sos: int = -1,
               eos: int = -1,
               bptt_size: int = 100,
               distributed: bool = False,
               kaldi_format: bool = True,
               min_token_num: int = 2,
               max_token_num: int = 2000,
               max_batch_size: int = 64,
               num_workers: int = 0) -> Iterable[Dict]:
    """
    The BPTT dataloader for LM training
    Args:
        text: path of the text/token file
        vocab_dict: vocabulary dictionary
        sos|eos: sos|eos ID
        distributed: for distributed training or not
        kaldi_format: whether text/token file is in kaldi format
        train: in training mode or not
        {min|max}_token_num: boundary of the token length
        max_batch_size: in this case equals to #batch_size
        min_batch_size: not used here
        num_workers: number workers used in dataloader, not used here
    """
    return BpttDataloader(Dataset(text, vocab_dict, kaldi_format=kaldi_format),
                          max_batch_size,
                          bptt_size=bptt_size,
                          sos=sos,
                          eos=eos,
                          shuffle=train,
                          distributed=distributed,
                          min_token_num=min_token_num,
                          max_token_num=max_token_num)


class SequenceSampler(dat.Sampler):
    """
    A custom sequence sampler for BPTT dataloader
    Args:
        dataset: instance of dat.Dataset
        shuffle: shuffle batches or not
        distributed: in distributed mode or not
        {min|max}_token_num: boundary of the token length
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 shuffle: bool = False,
                 distributed: bool = False,
                 min_token_num: int = 2,
                 max_token_num: int = 2000) -> None:
        if distributed:
            self.world_size = dist.world_size()
            self.header = f"SequenceSampler (rank {dist.rank()})"
        else:
            self.world_size = 1
            self.header = "SequenceSampler"
        logger.info(f"{self.header}: filtering utterances ...")
        self.indices = filter_utts(dataset,
                                   min_token_num=min_token_num,
                                   max_token_num=max_token_num)
        kept_utt_num = len(self.indices)
        self.epoch = 0
        self.batches = list(range(kept_utt_num))
        self.shuffle = shuffle
        self.distributed = distributed
        self.num_batches = kept_utt_num // self.world_size

    def __iter__(self) -> Iterator[List[int]]:
        indices = derive_indices(self.num_batches,
                                 seed=self.epoch,
                                 shuffle=self.shuffle,
                                 distributed=self.distributed)
        indices = [self.indices[i] for i in indices]
        return iter(indices)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches


class BpttDataloader(object):
    """
    The BPTT dataloader
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 batch_size: int,
                 bptt_size: int = 100,
                 sos: int = -1,
                 eos: int = -1,
                 shuffle: bool = True,
                 distributed: bool = False,
                 min_token_num: int = 2,
                 max_token_num: int = 2000) -> None:
        if sos < 0 or eos < 0:
            raise ValueError(f"Invalid sos/eos value: {sos}/{eos}")
        self.eos = eos
        self.sos = sos
        self.bptt_size = bptt_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.sampler = SequenceSampler(dataset,
                                       shuffle=shuffle,
                                       distributed=distributed,
                                       min_token_num=min_token_num,
                                       max_token_num=max_token_num)

    def __iter__(self) -> Iterator[Dict]:
        # B x N
        # TODO: may be slow when dataset is large
        batch = concat_data(self.batch_size,
                            self.dataset,
                            self.sampler,
                            sos=self.sos,
                            eos=self.eos)
        for t in range(0, batch.shape[-1], self.bptt_size):
            if t + 1 + self.bptt_size > batch.shape[-1]:
                break
            yield {
                "#utt": self.batch_size,
                "#tok": self.batch_size * self.bptt_size,
                "src": batch[:, t:t + self.bptt_size],
                "tgt": batch[:, t + 1:t + 1 + self.bptt_size],
                "reset": t == 0
            }

    def __len__(self) -> int:
        return 0

    def set_epoch(self, epoch: int) -> NoReturn:
        self.sampler.set_epoch(epoch)
