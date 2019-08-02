#!/usr/bin/env python

# wujian@2019

import random

import numpy as np
import torch as th
import torch.utils.data as dat

from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from kaldi_python_io import Reader as BaseReader
from kaldi_python_io import ScriptReader
from kaldi_python_io.functional import read_kaldi_mat


def make_dataloader(train=True,
                    feats_scp="",
                    token_scp="",
                    gcmvn="",
                    norm_means=True,
                    norm_vars=True,
                    utt2num_frames="",
                    max_token_num=400,
                    max_frame_num=3000,
                    min_frame_num=40,
                    adapt_frame_num=800,
                    adapt_token_num=150,
                    batch_size=32,
                    num_workers=0,
                    min_batch_size=4):
    dataset = Dataset(feats_scp,
                      token_scp,
                      utt2num_frames,
                      gcmvn=gcmvn,
                      norm_means=True,
                      norm_vars=True,
                      max_token_num=max_token_num,
                      max_frame_num=max_frame_num,
                      min_frame_num=min_frame_num)
    return AmDataLoader(dataset,
                        shuffle=train,
                        num_workers=num_workers,
                        adapt_frame_num=adapt_frame_num,
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
        self.token_reader = self._process(token_scp,
                                          utt2num_frames,
                                          max_token_num=max_token_num,
                                          max_frame_num=max_frame_num,
                                          min_frame_num=min_frame_num)

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        key = tok["key"]
        feats = self.cmvn(self.feats_reader[key])
        return {
            "dur": tok["dur"],
            "len": tok["len"],
            "feats": feats,
            "token": tok["tok"]
        }

    def __len__(self):
        return len(self.token_reader)

    def _process(self,
                 token_scp,
                 utt2num_frames,
                 max_token_num=400,
                 max_frame_num=3000,
                 min_frame_num=40):
        utt2num_frames = BaseReader(utt2num_frames, value_processor=int)
        token_reader = BaseReader(
            token_scp,
            value_processor=lambda l: [int(n) for n in l],
            num_tokens=-1)
        token_set = []
        for key, token in token_reader:
            L = len(token)
            if L > max_token_num:
                continue
            num_frames = utt2num_frames[key]
            if num_frames < min_frame_num or num_frames > max_frame_num:
                continue
            token_set.append({
                "key": key,
                "dur": num_frames,
                "tok": token,
                "len": L
            })
        # long -> short
        token_set = sorted(token_set, key=lambda d: d["dur"], reverse=True)
        return token_set


class BatchSampler(dat.Sampler):
    """
    A customer batchsampler
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 adapt_frame_num=800,
                 adapt_token_num=150,
                 min_batch_size=4):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_idx = self._work_batch_index(adapt_frame_num,
                                                adapt_token_num,
                                                batch_size,
                                                min_batch_size=min_batch_size)

    def _work_batch_index(self,
                          adapt_frame_num,
                          adapt_token_num,
                          batch_size,
                          min_batch_size=4):
        beg = 0
        tot = len(self.dataset)
        cur_bz = batch_size
        idx_bz = []
        while beg + cur_bz <= tot:
            cur = self.dataset.token_reader[beg]
            cur_ilen = cur["dur"]
            cur_olen = cur["len"]
            factor = max(cur_ilen // adapt_frame_num,
                         cur_olen // adapt_token_num)
            cur_bz = int(max(min_batch_size, batch_size // (1 + factor)))
            idx_bz.append((beg, beg + cur_bz))
            beg += cur_bz
        return idx_bz

    def __iter__(self):
        n = len(self.batch_idx)
        idx_order = th.randperm(n) if self.shuffle else th.arange(n)
        for idx in idx_order.tolist():
            beg, end = self.batch_idx[idx]
            yield range(beg, end)

    def __len__(self):
        return len(self.batch_idx)


def egs_collate(egs):
    return {
        "x_pad":
        pad_sequence([th.from_numpy(eg["feats"]) for eg in egs],
                     batch_first=True,
                     padding_value=0),
        "y_pad":
        pad_sequence([th.as_tensor(eg["token"]) for eg in egs],
                     batch_first=True,
                     padding_value=-1),
        "x_len":    # N, number of the frames
        th.tensor([eg["dur"] for eg in egs], dtype=th.int64),
        "y_len":    # N, length of the tokens
        th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }


class AmDataLoader(object):
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
                               adapt_frame_num=800,
                               adapt_token_num=150,
                               min_batch_size=4)
        self.batch_loader = dat.DataLoader(dataset,
                                           batch_sampler=sampler,
                                           num_workers=num_workers,
                                           collate_fn=egs_collate)

    def __len__(self):
        return len(self.batch_loader)

    def __iter__(self):
        for egs in self.batch_loader:
            yield egs