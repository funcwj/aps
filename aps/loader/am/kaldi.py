#!/usr/bin/env python

# wujian@2019
"""
Dataloader for kaldi features
"""
import random

import numpy as np
import torch as th
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from kaldi_python_io import ScriptReader

from aps.loader.am.utils import process_token, BatchSampler


def DataLoader(train=True,
               distributed=False,
               feats_scp="",
               token="",
               utt2dur="",
               max_token_num=400,
               max_dur=3000,
               min_dur=40,
               adapt_dur=800,
               adapt_token_num=150,
               batch_size=32,
               num_workers=0,
               min_batch_size=4):
    dataset = Dataset(feats_scp,
                      token,
                      utt2dur,
                      max_token_num=max_token_num,
                      max_frame_num=max_dur,
                      min_frame_num=min_dur)
    return KaldiDataLoader(dataset,
                           shuffle=train,
                           distributed=distributed,
                           num_workers=num_workers,
                           adapt_frame_num=adapt_dur,
                           adapt_token_num=adapt_token_num,
                           batch_size=batch_size,
                           min_batch_size=min_batch_size)


class Dataset(dat.Dataset):
    """
    Dataset for kaldi features
    """
    def __init__(self,
                 feats_scp,
                 token,
                 utt2num_frames,
                 max_token_num=400,
                 max_frame_num=3000,
                 min_frame_num=40):
        self.feats_reader = ScriptReader(feats_scp)
        # sorted
        self.token_reader = process_token(token,
                                          utt2num_frames,
                                          max_token_num=max_token_num,
                                          max_dur=max_frame_num,
                                          min_dur=min_frame_num)

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        key = tok["key"]
        return {
            "dur": tok["dur"],
            "len": tok["len"],
            "feats": self.feats_reader[key],
            "token": tok["tok"]
        }

    def __len__(self):
        return len(self.token_reader)


def egs_collate(egs):
    def pad_seq(olist, value=0):
        return pad_sequence(olist, batch_first=True, padding_value=value)

    return {
        "src_pad":  # N x S
        pad_seq([th.from_numpy("feats") for eg in egs], value=0),
        "tgt_pad":  # N x T
        pad_seq([th.as_tensor(eg["token"]) for eg in egs], value=-1),
        "src_len":  # N, number of the frames
        th.tensor([eg["dur"] for eg in egs], dtype=th.int64),
        "tgt_len":  # N, length of the tokens
        th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }


class KaldiDataLoader(object):
    """
    Acoustic dataloader for seq2seq model training
    """
    def __init__(self,
                 dataset,
                 shuffle=True,
                 distributed=False,
                 num_workers=0,
                 adapt_frame_num=800,
                 adapt_token_num=150,
                 batch_size=32,
                 min_batch_size=4):
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
                               distributed=distributed,
                               adapt_dur=adapt_frame_num,
                               adapt_token_num=adapt_token_num,
                               min_batch_size=min_batch_size)
        self.batch_loader = dat.DataLoader(dataset,
                                           batch_sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=egs_collate)

    def __len__(self):
        return len(self.batch_loader)

    def __iter__(self):
        for egs in self.batch_loader:
            yield egs