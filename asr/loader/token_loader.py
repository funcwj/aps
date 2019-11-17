#!/usr/bin/env python

# wujian@2019
"""
TODO: dataloader for RNNLM
"""

import random
import torch as th

import torch.utils.data as dat

from kaldi_python_io import Reader as BaseReader


def make_token_loader(train=True,
                      eos=-1,
                      token="",
                      batch_size=64,
                      chunk_size=20):
    dataset = Dataset(token, eos=eos)
    return DataLoader(dataset,
                      shuffle=train,
                      batch_size=batch_size,
                      chunk_size=chunk_size)


class Dataset(dat.Dataset):
    """
    Dataset for text corpus
    """
    def __init__(self, token, eos=-1):
        if eos < 0:
            raise ValueError(f"Invalid EOS value: {eos}")
        token_reader = BaseReader(
            token,
            value_processor=lambda l: [int(n) for n in l],
            num_tokens=-1)
        self.token_list = []
        for key, tok in token_reader:
            if not len(tok):
                raise RuntimeError(f"Empty utterance: {key}")
            self.token_list.append(tok + [eos])

    def __getitem__(self, index):
        return self.token_list[index]

    def __len__(self):
        return len(self.token_list)


class DataLoader(object):
    """
    LM loader for bptt training
    """
    def __init__(self, dataset, shuffle=True, batch_size=64, chunk_size=20):
        utt_lens = [len(tok) for tok in dataset]
        seq_lens = sum(utt_lens) // batch_size
        self.num_batches = seq_lens // chunk_size - 1
        self.shuffle = shuffle
        self.token_list = [th.tensor(tok, dtype=th.int64) for tok in dataset]
        self.batch_size = batch_size
        self.chunk_size = chunk_size

    def batchify(self):
        if self.shuffle:
            # shuffle the token list
            random.shuffle(self.token_list)
        # list => tensor
        token_set = th.cat(self.token_list)
        S = token_set.size(0) // self.batch_size
        # NS
        token_set = token_set[:S * self.batch_size]
        # N x S
        return token_set.view(self.batch_size, S).contiguous()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # N x S
        token_batch = self.batchify()
        for beg in range(0, token_batch.shape[-1] - self.chunk_size,
                         self.chunk_size):
            end = beg + self.chunk_size
            yield {
                "x": token_batch[:, beg:end],
                "y": token_batch[:, beg + 1:end + 1]
            }
