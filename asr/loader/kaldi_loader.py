#!/usr/bin/env python

# wujian@2019

import random

import numpy as np
import torch as th
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate

from kaldi_python_io import ScriptReader
from kaldi_python_io.functional import read_kaldi_mat

from .utils import process_token, BatchSampler


def make_dataloader(train=True,
                    feats_scp="",
                    token_scp="",
                    gcmvn="",
                    norm_means=True,
                    norm_vars=True,
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
                      token_scp,
                      utt2dur,
                      gcmvn=gcmvn,
                      norm_means=True,
                      norm_vars=True,
                      max_token_num=max_token_num,
                      max_frame_num=max_dur,
                      min_frame_num=min_dur)
    return DataLoader(dataset,
                      shuffle=train,
                      num_workers=num_workers,
                      adapt_frame_num=adapt_dur,
                      adapt_token_num=adapt_token_num,
                      batch_size=batch_size,
                      min_batch_size=min_batch_size)


class Cmvn(object):
    """
    Class to do global cmvn or utterance-level cmvn
    """
    def __init__(self, gcmvn="", norm_means=True, norm_vars=True):
        if gcmvn:
            self.gmean, self.gstd = self._load_cmvn(gcmvn)
        else:
            self.gmean, self.gstd = None, None
        self.norm_means = norm_means
        self.norm_vars = norm_vars

    def _load_cmvn(self, cmvn_mat):
        """
        Compute mean/std from cmvn.mat
        """
        cmvn = read_kaldi_mat(cmvn_mat)
        N = cmvn[0, -1]
        mean = cmvn[0, :-1] / N
        var = cmvn[1, :-1] / N - mean**2
        return mean, var**0.5

    def __call__(self, mat):
        if self.norm_means:
            mat -= (self.gmean if self.gmean is not None else np.mean(mat, 0))
        if self.norm_vars:
            mat -= (self.gstd if self.gstd is not None else np.std(mat, 0))
        return mat


class Dataset(dat.Dataset):
    """
    Dataset for kaldi features
    """
    def __init__(self,
                 feats_scp,
                 token_scp,
                 utt2num_frames,
                 gcmvn="",
                 norm_means=True,
                 norm_vars=True,
                 max_token_num=400,
                 max_frame_num=3000,
                 min_frame_num=40):
        self.feats_reader = ScriptReader(feats_scp)
        self.cmvn = Cmvn(gcmvn=gcmvn,
                         norm_means=norm_means,
                         norm_vars=norm_vars)
        # sorted
        self.token_reader = process_token(token_scp,
                                          utt2num_frames,
                                          max_token_num=max_token_num,
                                          max_dur=max_frame_num,
                                          min_dur=min_frame_num)

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        key = tok["key"]
        feats = self.feats_reader[key]
        return {
            "dur": tok["dur"],
            "len": tok["len"],
            "feats": self.cmvn(feats),
            "token": tok["tok"]
        }

    def __len__(self):
        return len(self.token_reader)


def egs_collate(egs):
    def pad_seq(olist, value=0):
        return pad_sequence(olist, batch_first=True, padding_value=value)

    return {
        "x_pad":  # N x S
        pad_seq([th.from_numpy("feats") for eg in egs], value=0),
        "y_pad":  # N x T
        pad_seq([th.as_tensor(eg["token"]) for eg in egs], value=-1),
        "x_len":  # N, number of the frames
        th.tensor([eg["dur"] for eg in egs], dtype=th.int64),
        "y_len":  # N, length of the tokens
        th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }


class DataLoader(object):
    """
    Acoustic dataloader for seq2seq model training
    """
    def __init__(self,
                 dataset,
                 shuffle=True,
                 num_workers=0,
                 adapt_frame_num=800,
                 adapt_token_num=150,
                 batch_size=32,
                 min_batch_size=4):
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
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