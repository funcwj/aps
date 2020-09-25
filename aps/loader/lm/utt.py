#!/usr/bin/env python

# wujian@2019
"""
for RNNLM (utterance corpus)
"""
import random
import warnings
import torch as th

import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from kaldi_python_io import Reader as BaseReader


def DataLoader(text="",
               vocab_dict="",
               train=True,
               sos=-1,
               eos=-1,
               min_token_num=2,
               max_token_num=2000,
               adapt_token_num=400,
               min_batch_size=8,
               batch_size=64,
               num_workers=0):
    dataset = Dataset(text,
                      vocab_dict,
                      min_token_num=min_token_num,
                      max_token_num=max_token_num)
    return UttDataLoader(dataset,
                         sos=sos,
                         eos=eos,
                         shuffle=train,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         min_batch_size=min_batch_size,
                         adapt_token_num=adapt_token_num)


class BatchSampler(dat.Sampler):
    """
    A custom batchsampler
    """

    def __init__(self,
                 token_set,
                 batch_size,
                 min_batch_size=8,
                 adapt_token_num=400,
                 shuffle=False):
        self.min_batch_size = min_batch_size
        self.adapt_token_num = adapt_token_num
        self.genfunc = th.randperm if shuffle else th.arange
        self.batches = []
        chunk_size = 10000 * batch_size
        for i in range(0, len(token_set), chunk_size):
            indices = self._sort_indices(token_set[i:i + chunk_size],
                                         batch_size)
            self.batches += indices

    def _sort_indices(self, subset, batch_size):
        utts_len = [len(seq) for seq in subset]
        # short -> long
        desc_idx = th.argsort(th.tensor(utts_len, dtype=th.int32),
                              descending=False).tolist()
        batches = []
        beg, cur_bz = 0, batch_size
        while beg + self.min_batch_size <= len(desc_idx):
            cur_len = utts_len[desc_idx[beg]]
            factor = cur_len // self.adapt_token_num
            cur_bz = int(max(self.min_batch_size, batch_size // (1 + factor)))
            batches.append(desc_idx[beg:beg + cur_bz])
            beg += cur_bz
        return batches

    def __iter__(self):
        indices = self.genfunc(len(self.batches)).tolist()
        for i in indices:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


class Dataset(dat.Dataset):
    """
    Dataset for token corpus
    """

    def __init__(self,
                 text,
                 vocab_dict,
                 min_token_num=2,
                 max_token_num=2000,
                 eos=None):
        if vocab_dict:
            text_reader = BaseReader(text, num_tokens=-1, restrict=False)
        else:
            text_reader = BaseReader(
                text,
                value_processor=lambda tok: list(map(int, tok)),
                num_tokens=-1,
                restrict=False)
        self.token_set = []
        for key, tokens in text_reader:
            if len(tokens) < min_token_num:
                warnings.warn(f"Pass utterance that is too short: {key}")
            elif len(tokens) > max_token_num:
                warnings.warn(f"Pass utterance that is too long: {key}")
            else:
                if vocab_dict:
                    toks = []
                    for t in tokens:
                        toks.append(vocab_dict[t] if t in
                                    vocab_dict else vocab_dict["<unk>"])
                else:
                    toks = tokens
                if eos is None:
                    self.token_set.append(toks)
                else:
                    self.token_set.append(toks + [eos])

    def __getitem__(self, index):
        return self.token_set[index]

    def __len__(self):
        return len(self.token_set)


class UttDataLoader(object):
    """
    DataLoader for LM training
    """

    def __init__(self,
                 dataset,
                 sos=-1,
                 eos=-1,
                 shuffle=True,
                 batch_size=64,
                 num_workers=0,
                 adapt_token_num=400,
                 min_batch_size=8):
        if sos < 0 or eos < 0:
            raise ValueError(f"Invalid sos/eos value: {sos}/{eos}")
        self.eos = eos
        self.sos = sos
        sampler = BatchSampler(dataset.token_set,
                               batch_size,
                               shuffle=shuffle,
                               min_batch_size=min_batch_size,
                               adapt_token_num=adapt_token_num)
        self.batch_loader = dat.DataLoader(dataset,
                                           batch_sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=self.egs_collate)

    def egs_collate(self, egs):
        ntok = [len(eg) + 1 for eg in egs]
        sos_egs = [th.as_tensor([self.sos] + eg) for eg in egs]
        egs_eos = [th.as_tensor(eg + [self.eos]) for eg in egs]
        return {
            "src": pad_sequence(sos_egs, batch_first=True, padding_value=0),
            "tgt": pad_sequence(egs_eos, batch_first=True, padding_value=-1),
            "len": th.tensor(ntok, dtype=th.int64)
        }

    def __len__(self):
        return len(self.batch_loader)

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        for egs in self.batch_loader:
            yield egs
