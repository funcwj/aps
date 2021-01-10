#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
for RNNLM (utterance corpus)
"""
import gzip
import warnings
import numpy as np
import torch as th

import torch.utils.data as dat
import aps.distributed as dist

from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List, Dict, Optional, Iterator, Iterable
from aps.libs import ApsRegisters
from aps.const import IGNORE_ID, UNK_TOKEN


@ApsRegisters.loader.register("lm_utt")
def DataLoader(text: str = "",
               vocab_dict: Optional[Dict] = None,
               train: bool = True,
               sos: int = -1,
               eos: int = -1,
               distributed: bool = False,
               kaldi_format: bool = True,
               chunk_size_for_sort: int = 10000,
               min_token_num: int = 2,
               max_token_num: int = 2000,
               adapt_token_num: int = 400,
               min_batch_size: int = 8,
               max_batch_size: int = 64,
               num_workers: int = 0) -> Iterable[Dict]:
    """
    The utterance-level dataloader for LM training
    Args:
        text: path of the text/token file
        vocab_dict: vocabulary dictionary
        sos|eos: sos|eos ID
        distributed: for distributed training or not
        kaldi_format: whether text/token file is in kaldi format
        train: in training mode or not
        max_batch_size: maximum value of #batch_size
        min_batch_size: minimum value of #batch_size
        num_workers: number workers used in dataloader
        {min|max}_token_num: boundary of the token length
        adapt_token_num: used for #batch_size reduction
        chunk_size_for_sort: #chunk_size for mini-batch sorting, we perform sort
                             in each chunk (because LM corpus may very big)
    """
    return UttDataLoader(Dataset(text, vocab_dict, kaldi_format=kaldi_format),
                         sos=sos,
                         eos=eos,
                         shuffle=train,
                         max_batch_size=max_batch_size,
                         distributed=distributed,
                         num_workers=num_workers,
                         min_token_num=min_token_num,
                         max_token_num=max_token_num,
                         min_batch_size=min_batch_size,
                         adapt_token_num=adapt_token_num,
                         chunk_size_for_sort=chunk_size_for_sort)


class Dataset(dat.Dataset):
    """
    Dataset for text corpus
    Args:
        text: path of the text/token file
        vocab_dict: vocabulary dictionary
        kaldi_format: whether text/token file is in kaldi format
    """

    def __init__(self,
                 text: str,
                 vocab_dict: Optional[Dict],
                 kaldi_format: bool = True) -> None:
        self.vocab = vocab_dict
        self.kaldi_format = kaldi_format
        self.token = self._load(text)

    def _load(self, text):
        """
        Read all the lines in text file
        """
        if text[-3:] == ".gz":
            with gzip.open(text, "r") as gzip_f:
                token = [line.decode() for line in gzip_f.readlines()]
        else:
            with open(text, "r") as text_f:
                token = text_f.readlines()
        return token

    def __getitem__(self, index: int) -> List[int]:
        str_toks = self.token[index].split()
        # remove the first token (key)
        if self.kaldi_format:
            str_toks = str_toks[1:]
        if self.vocab:
            int_toks = [
                (self.vocab[t] if t in self.vocab else self.vocab[UNK_TOKEN])
                for t in str_toks
            ]
        else:
            int_toks = list(map(int, str_toks))
        return int_toks

    def __len__(self) -> int:
        return len(self.token)


class BatchSampler(dat.Sampler):
    """
    A custom batch sampler for LM dataset
    Args:
        dataset: instance of dat.Dataset
        max_batch_size: maximum value of #batch_size
        shuffle: shuffle batches or not
        distributed: in distributed mode or not
        {min|max}_token_num: boundary of the token length
        min_batch_size: minimum value of #batch_size
        adapt_token_num: used for #batch_size reduction
        chunk_size_for_sort: we perform sort in each chunk (for large LM corpus)
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 max_batch_size: int,
                 shuffle: bool = False,
                 distributed: bool = False,
                 min_token_num: int = 2,
                 max_token_num: int = 2000,
                 min_batch_size: int = 8,
                 adapt_token_num: int = 400,
                 chunk_size_for_sort: int = 10000) -> None:
        self.distributed = distributed
        if distributed:
            self.world_size = dist.world_size()
            self.rank = dist.rank()
            self.header = f"BatchSampler (rank {self.rank})"
        else:
            self.header = "BatchSampler"
        self.const = {
            "min": min_token_num,
            "max": max_token_num,
            "floor": min_batch_size,
            "adapt": adapt_token_num
        }
        self.genfunc = th.randperm if shuffle else th.arange
        self.batches = []
        chunk_size = chunk_size_for_sort
        kept_index = self._filter_indices(dataset)
        total_utts = len(kept_index)
        num_parts = total_utts // chunk_size + 1
        print(f"{self.header}: sort indices ({num_parts} parts) ...",
              flush=True)
        for base in range(0, total_utts, chunk_size):
            indices = self._sort_indices(dataset, [
                kept_index[i]
                for i in range(base, min(base + chunk_size, total_utts))
            ], max_batch_size)
            self.batches += indices
            done = min(base + chunk_size, total_utts) * 100 / float(total_utts)
            print(f"{self.header}: done {done:.2f}% ...", flush=True)
        if distributed:
            self.num_batches = len(self.batches) // self.world_size
        else:
            self.num_batches = len(self.batches)
        self.epoch = 0

    def _filter_indices(self, dataset: dat.Dataset) -> List[int]:
        """
        Return utterance index used for training (pass short/long utterances)
        """
        print(f"{self.header}: filtering indices ...", flush=True)
        kept_index = []
        filter_sutt, filter_lutt = 0, 0
        for i in range(len(dataset)):
            tok_len = len(dataset[i])
            if tok_len < self.const["min"]:
                filter_sutt += 1
            elif tok_len > self.const["max"]:
                filter_lutt += 1
            else:
                kept_index.append(i)
        if filter_lutt or filter_sutt:
            ratio_lutt = filter_lutt * 100.0 / len(dataset)
            ratio_sutt = filter_sutt * 100.0 / len(dataset)
            warnings.warn(
                f"{self.header}: filter {ratio_lutt:.2f}/{ratio_sutt:.2f}% " +
                "long/short utterances...")
        return kept_index

    def _sort_indices(self, dataset: dat.Dataset, subset: List[int],
                      max_batch_size: int) -> List[List[int]]:
        """
        Sort mini-batches in each subset
        """
        toks_len = [len(dataset[i]) for i in subset]
        # long -> short
        sort_idx = np.argsort(toks_len)[::-1]
        batches = []
        beg, cur_bz = 0, max_batch_size
        while beg + cur_bz <= len(sort_idx):
            cur_len = toks_len[sort_idx[beg]]
            factor = (cur_len - 1) // self.const["adapt"]
            cur_bz = int(
                max(self.const["floor"], max_batch_size // (1 + factor)))
            batches.append([subset[i] for i in sort_idx[beg:beg + cur_bz]])
            beg += cur_bz
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self.distributed:
            # deterministically shuffle based on epoch
            g = th.Generator()
            g.manual_seed(self.epoch)
            N = self.num_batches * self.world_size
            indices = th.randperm(N, generator=g).tolist()
            indices = indices[self.rank:N:self.world_size]
        else:
            indices = self.genfunc(self.num_batches).tolist()
        indices = [self.batches[i] for i in indices]
        return iter(indices)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches


class UttDataLoader(dat.DataLoader):
    """
    The utterance level dataLoader for LM training
    Args:
        dataset: instance of dat.Dataset
        sos|eos: sos|eos ID
        shuffle: shuffle batches or not
        max_batch_size: maximum value of #batch_size
        num_workers: number workers used in dataloader
        min|max_token_num: boundary of the token length
        min_batch_size: minimum value of #batch_size
        adapt_token_num: used for #batch_size reduction
        chunk_size_for_sort: #chunk_size for mini-batch sorting, we perform sort
                             in each chunk (for large LM corpus)
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 sos: int = -1,
                 eos: int = -1,
                 shuffle: bool = True,
                 max_batch_size: int = 64,
                 distributed: bool = False,
                 num_workers: int = 0,
                 min_token_num: int = 2,
                 max_token_num: int = 2000,
                 adapt_token_num: int = 400,
                 min_batch_size: int = 8,
                 chunk_size_for_sort: int = 1000) -> None:
        if sos < 0 or eos < 0:
            raise ValueError(f"Invalid sos/eos value: {sos}/{eos}")
        self.eos = eos
        self.sos = sos
        sampler = BatchSampler(dataset,
                               max_batch_size,
                               shuffle=shuffle,
                               distributed=distributed,
                               min_token_num=min_token_num,
                               max_token_num=max_token_num,
                               min_batch_size=min_batch_size,
                               adapt_token_num=adapt_token_num,
                               chunk_size_for_sort=chunk_size_for_sort)
        super(UttDataLoader, self).__init__(dataset,
                                            batch_sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.egs_collate)

    def egs_collate(self, egs):
        sos_egs = [th.as_tensor([self.sos] + eg) for eg in egs]
        egs_eos = [th.as_tensor(eg + [self.eos]) for eg in egs]
        return {
            "#utt":
                len(egs),
            "#tok":
                sum([len(eg) + 1 for eg in egs]),
            "src":
                pad_sequence(sos_egs, batch_first=True, padding_value=self.eos),
            "tgt":
                pad_sequence(egs_eos, batch_first=True,
                             padding_value=IGNORE_ID),
            "len":
                th.tensor([len(eg) + 1 for eg in egs], dtype=th.int64)
        }

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
