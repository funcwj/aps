#!/usr/bin/env python

# wujian@2019

import torch as th

import torch.utils.data as dat
# import torch.distributed as dist
import aps.distributed as dist

from kaldi_python_io import Reader as BaseReader


def process_token(text,
                  utt2dur,
                  vocab_dict,
                  max_token_num=400,
                  min_token_num=2,
                  max_dur=3000,
                  min_dur=40):
    utt2dur = BaseReader(utt2dur, value_processor=float)
    if vocab_dict:
        text_reader = BaseReader(text, num_tokens=-1, restrict=False)
    else:
        text_reader = BaseReader(
            text,
            value_processor=lambda tok: list(map(int, tok)),
            num_tokens=-1,
            restrict=False)
    token_set = []
    for key, tokens in text_reader:
        L = len(tokens)
        if L > max_token_num or L <= min_token_num:
            continue
        if key not in utt2dur:
            continue
        num_frames = utt2dur[key]
        if num_frames < min_dur or num_frames > max_dur:
            continue
        stats = {"key": key, "dur": num_frames, "len": L}
        if vocab_dict:
            toks = []
            for t in tokens:
                toks.append(vocab_dict[t] if t in
                            vocab_dict else vocab_dict["<unk>"])
            stats["tok"] = toks
        else:
            stats["tok"] = tokens
        token_set.append(stats)
    # long -> short
    token_set = sorted(token_set, key=lambda d: d["dur"], reverse=True)
    N = len(token_set)
    if N < 10:
        raise RuntimeError(
            f"Too less utterances: {N}, check data configurations")
    return token_set


class BatchSampler(dat.Sampler):
    """
    A custom batchsampler
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 batch_mode="adaptive",
                 adapt_dur=800,
                 adapt_token_num=150,
                 min_batch_size=4,
                 distributed=False):
        if batch_mode not in ["adaptive", "constraint"]:
            raise ValueError(f"Unsupported batch mode: {batch_mode}")
        self.distributed = distributed
        if distributed:
            self.world_size = dist.world_size()
            self.rank = dist.rank()
        if batch_mode == "adaptive":
            batches = self._work_adapt_batch_index(
                dataset,
                adapt_dur,
                adapt_token_num,
                batch_size,
                min_batch_size=min_batch_size)
        else:
            batches = self._work_const_batch_index(dataset, batch_size)
        if distributed:
            self.num_batches = len(batches) // self.world_size
        else:
            self.num_batches = len(batches)
        self.batches = batches
        self.genfunc = th.randperm if shuffle else th.arange
        self.epoch = 0

    def _work_const_batch_index(self, dataset, batch_size):
        beg = 0
        tot = len(dataset)
        cur_dur = 0
        idx_bz = []
        # long -> short
        for idx in range(tot):
            cur = dataset.token_reader[idx]
            if idx == 0:
                if cur["dur"] > batch_size:
                    raise ValueError("batch_size is smaller than maximum "
                                     "length of the utterances")
            utt_dur = cur["dur"]
            if cur_dur < batch_size:
                cur_dur += utt_dur
            else:
                idx_bz.append((beg, idx))
                cur_dur = utt_dur
                beg = idx
        if tot - beg > 1:
            idx_bz.append((beg, tot))
        return idx_bz

    def _work_adapt_batch_index(self,
                                dataset,
                                adapt_dur,
                                adapt_token_num,
                                batch_size,
                                min_batch_size=4):
        beg = 0
        tot = len(dataset)
        cur_bz = batch_size
        idx_bz = []
        while beg + cur_bz <= tot:
            cur = dataset.token_reader[beg]
            cur_ilen = cur["dur"]
            cur_olen = cur["len"]
            factor = max(cur_ilen // adapt_dur, cur_olen // adapt_token_num)
            cur_bz = int(max(min_batch_size, batch_size // (1 + factor)))
            idx_bz.append((beg, beg + cur_bz))
            beg += cur_bz
        return idx_bz

    def __iter__(self):
        if self.distributed:
            # deterministically shuffle based on epoch
            g = th.Generator()
            g.manual_seed(self.epoch)
            N = self.num_batches * self.world_size
            indices = th.randperm(N, generator=g).tolist()
            indices = indices[self.rank:N:self.world_size]
        else:
            indices = self.genfunc(self.num_batches).tolist()
        for i in indices:
            yield list(range(*self.batches[i]))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches
