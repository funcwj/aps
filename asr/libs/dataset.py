#!/usr/bin/env python

# wujian@2019

import random

import torch as th

from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pad_sequence

from kaldi_python_io import Reader as BaseReader
from kaldi_python_io import ScriptReader


def make_dataloader(train=True,
                    feats_scp="",
                    token_scp="",
                    utt2num_frames="",
                    max_token_num=400,
                    max_frame_num=3000,
                    min_frame_num=40,
                    adapt_frame_num=800,
                    adapt_token_num=150,
                    batch_size=32,
                    min_batch_size=4):
    dataset = Dataset(feats_scp,
                      token_scp,
                      utt2num_frames,
                      max_token_num=max_token_num,
                      max_frame_num=max_frame_num,
                      min_frame_num=min_frame_num)
    return AmDataLoader(dataset,
                        shuffle=train,
                        adapt_frame_num=adapt_frame_num,
                        adapt_token_num=adapt_token_num,
                        batch_size=batch_size,
                        min_batch_size=min_batch_size)


class Dataset(object):
    def __init__(self,
                 feats_scp,
                 token_scp,
                 utt2num_frames,
                 max_token_num=400,
                 max_frame_num=3000,
                 min_frame_num=40):
        self.feats_reader = ScriptReader(feats_scp)
        # sorted
        self.token_reader = self._process(token_scp, utt2num_frames,
                                          max_token_num, max_frame_num,
                                          min_frame_num)

    def _get_egs(self, tok):
        return {
            "dur": tok["dur"],
            "len": tok["len"],
            "feats": self.feats_reader[tok["key"]],
            "token": tok["tok"]
        }

    def __iter__(self):
        for tok in self.token_reader:
            yield self._get_egs(tok)

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        return self._get_egs(tok)

    def __len__(self):
        N = len(self.token_reader)
        return N

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


class AmDataLoader(object):
    def __init__(self,
                 dataset,
                 shuffle=True,
                 adapt_frame_num=800,
                 adapt_token_num=150,
                 batch_size=32,
                 min_batch_size=4):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_idx = self._work_batch_index(adapt_frame_num,
                                                adapt_token_num,
                                                batch_size,
                                                min_batch_size=min_batch_size)

    def __len__(self):
        return len(self.batch_idx)

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

    def _collate(self, egs):
        """
        Generate egs for nnet training
            x_pad, x_len, y_pad, y_len
        """
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

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batch_idx)
        for idx in self.batch_idx:
            batch = [self.dataset[i] for i in range(*idx)]
            yield self._collate(batch)