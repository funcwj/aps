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


def DataLoader(token="",
               train=True,
               sos=-1,
               eos=-1,
               min_token_num=2,
               batch_size=64,
               num_workers=0,
               drop_last=False):
    dataset = Dataset(token, min_token_num=min_token_num)
    return UttDataLoader(dataset,
                         sos=sos,
                         eos=eos,
                         shuffle=train,
                         drop_last=drop_last,
                         batch_size=batch_size,
                         num_workers=num_workers)


class BatchSampler(dat.Sampler):
    """
    A custom batchsampler
    """
    def __init__(self, token_set, batch_size, shuffle=False, drop_last=False):
        num_utts = len(token_set)
        len_utts = [len(tok) for tok in token_set]
        order_idx = th.argsort(th.tensor(len_utts, dtype=th.int32)).tolist()
        # reverse order
        order_idx = order_idx[::-1]
        batches = []
        for i in range(0, num_utts, batch_size):
            if i + batch_size > num_utts:
                batches.append(order_idx[i:])
            else:
                batches.append(order_idx[i:i + batch_size])
        if drop_last:
            batches = batches[:-1]
        self.shuffle = shuffle
        self.batches = batches
        self.num_batches = len(batches)

    def __iter__(self):
        if self.shuffle:
            indices = th.randperm(self.num_batches).tolist()
        else:
            indices = th.arange(self.num_batches).tolist()
        for i in indices:
            yield self.batches[i]

    def __len__(self):
        return self.num_batches


class Dataset(dat.Dataset):
    """
    Dataset for token corpus
    """
    def __init__(self, token_scp, min_token_num=2, eos=None):
        token_reader = BaseReader(
            token_scp,
            value_processor=lambda l: [int(n) for n in l],
            num_tokens=-1)
        self.token_set = []
        for _, tok in token_reader:
            if len(tok) <= min_token_num:
                # warnings.warn(f"Pass short utterances: {key}")
                pass
            else:
                if eos is None:
                    self.token_set.append(tok)
                else:
                    self.token_set.append(tok + [eos])

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
                 drop_last=False):
        if sos < 0 or eos < 0:
            raise ValueError(f"Invalid sos/eos value: {sos}/{eos}")
        self.eos = eos
        self.sos = sos
        sampler = BatchSampler(dataset.token_set,
                               batch_size,
                               shuffle=shuffle,
                               drop_last=drop_last)
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

    def __iter__(self):
        for egs in self.batch_loader:
            yield egs


def run():
    loader = token_loader(token="token",
                          sos=1,
                          eos=0,
                          train=False,
                          batch_size=32)
    for egs in loader:
        print(egs["len"])
        print(egs["tgt"])


if __name__ == "__main__":
    run()