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
               kaldi_format: bool = True,
               chunk_size_for_sort: int = 10000,
               min_token_num: int = 2,
               max_token_num: int = 2000,
               adapt_token_num: int = 400,
               min_batch_size: int = 8,
               batch_size: int = 64,
               num_workers: int = 0) -> Iterable[Dict]:
    return UttDataLoader(Dataset(text, vocab_dict, kaldi_format=kaldi_format),
                         sos=sos,
                         eos=eos,
                         shuffle=train,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         min_token_num=min_token_num,
                         max_token_num=max_token_num,
                         min_batch_size=min_batch_size,
                         adapt_token_num=adapt_token_num,
                         chunk_size_for_sort=chunk_size_for_sort)


class Dataset(dat.Dataset):
    """
    Dataset for text corpus
    """

    def __init__(self,
                 text: str,
                 vocab_dict: Optional[Dict],
                 kaldi_format=True) -> None:
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
    A custom batchsampler
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 batch_size: int,
                 chunk_size_for_sort: int = 10000,
                 min_token_num: int = 2,
                 max_token_num: int = 2000,
                 min_batch_size: int = 8,
                 adapt_token_num: int = 400,
                 shuffle: bool = False) -> None:
        self.const = {
            "min": min_token_num,
            "max": max_token_num,
            "adapt": adapt_token_num,
            "floor": min_batch_size
        }
        self.genfunc = th.randperm if shuffle else th.arange
        self.batches = []
        chunk_size = chunk_size_for_sort
        total_utts = len(dataset)
        num_parts = total_utts // chunk_size + 1
        print(f"BatchSampler: sort indices ({num_parts} parts) ...", flush=True)
        for i in range(0, total_utts, chunk_size):
            indices = self._sort_indices(
                [dataset[i] for i in range(i, min(i + chunk_size, total_utts))],
                batch_size, i)
            self.batches += indices
            done = min(i + chunk_size, total_utts) * 100 / float(total_utts)
            print(f"BatchSampler: done {done:.2f}% ...", flush=True)

    def _sort_indices(self, subset: List[int], batch_size: int,
                      base: int) -> List[List[int]]:
        # short -> long
        toks_len = [len(toks) for toks in subset]
        desc_idx = np.argsort(toks_len)
        kept_desc_idx = []
        for i in range(len(desc_idx)):
            tok_len = toks_len[desc_idx[i]]
            if self.const["min"] <= tok_len <= self.const["max"]:
                kept_desc_idx.append(i)
        pass_utts = len(desc_idx) - len(kept_desc_idx)
        if pass_utts:
            warnings.warn(f"Pass {pass_utts} long/short utterances...")
        batches = []
        beg, cur_bz = 0, batch_size
        while beg + cur_bz <= len(kept_desc_idx):
            cur_len = toks_len[kept_desc_idx[beg]]
            factor = cur_len // self.const["adapt"]
            cur_bz = int(max(self.const["floor"], batch_size // (1 + factor)))
            batches.append([base + i for i in kept_desc_idx[beg:beg + cur_bz]])
            beg += cur_bz
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = len(self.batches)
        indices = [self.batches[i] for i in self.genfunc(num_batches).tolist()]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.batches)


class UttDataLoader(dat.DataLoader):
    """
    DataLoader for LM training
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 sos: int = -1,
                 eos: int = -1,
                 shuffle: bool = True,
                 batch_size: int = 64,
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
                               batch_size,
                               shuffle=shuffle,
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
            "src":
                pad_sequence(sos_egs, batch_first=True, padding_value=self.eos),
            "tgt":
                pad_sequence(egs_eos, batch_first=True,
                             padding_value=IGNORE_ID),
            "len":
                th.tensor([len(eg) + 1 for eg in egs], dtype=th.int64)
        }

    def set_epoch(self, epoch: int) -> NoReturn:
        pass
