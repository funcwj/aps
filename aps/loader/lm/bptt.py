#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
for RNNLM (paragraph corpus)
ref: https://github.com/pytorch/examples/tree/master/word_language_model
"""

import random
import torch as th

import torch.utils.data as dat

from aps.loader.lm.utt import Dataset
from kaldi_python_io import Reader as BaseReader


def DataLoader(text="",
               vocab_dict="",
               train=True,
               sos=-1,
               eos=-1,
               faster=False,
               batch_size=64,
               chunk_size=20,
               num_workers=0,
               min_token_num=5):
    dataset = Dataset(text,
                      vocab_dict,
                      faster=faster,
                      eos=eos,
                      min_token_num=min_token_num)
    return BpttDataLoader(dataset,
                          shuffle=train,
                          batch_size=batch_size,
                          chunk_size=chunk_size)


class BpttDataLoader(object):
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

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        # N x S
        token_batch = self.batchify()
        for beg in range(0, token_batch.shape[-1] - self.chunk_size,
                         self.chunk_size):
            end = beg + self.chunk_size
            yield {
                "src": token_batch[:, beg:end],
                "tgt": token_batch[:, beg + 1:end + 1]
            }
