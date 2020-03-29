#!/usr/bin/env python

# wujian@2019

import subprocess
import torch as th

import torch.utils.data as dat
import torch.distributed as dist

from kaldi_python_io import Reader as BaseReader


def process_token(token,
                  utt2dur,
                  max_token_num=400,
                  min_token_num=2,
                  max_dur=3000,
                  min_dur=40):
    utt2dur = BaseReader(utt2dur, value_processor=float)
    token_reader = BaseReader(token,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1,
                              restrict=False)
    token_set = []
    for key, token in token_reader:
        L = len(token)
        if L > max_token_num or L <= min_token_num:
            continue
        if key not in utt2dur:
            continue
        num_frames = utt2dur[key]
        if num_frames < min_dur or num_frames > max_dur:
            continue
        token_set.append({
            "key": key,
            "dur": num_frames,
            "tok": token,
            "len": L
        })
    # long -> short
    token_set = sorted(token_set, key=lambda d: d["dur"], reverse=True)
    N = len(token_set)
    if N < 10:
        raise RuntimeError(
            f"Too less utterances: {N}, check data configurations")
    return token_set

def run_command(command, wait=True):
    """ 
    Runs shell commands. These are usually a sequence of 
    commands connected by pipes, so we use shell=True
    """
    p = subprocess.Popen(command,
                         shell=True,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode != 0:
            stderr_str = bytes.decode(stderr)
            raise Exception("There was an error while running the " +
                            f"command \"{command}\":\n{stderr_str}\n")
        return stdout, stderr
    else:
        return p


class BatchSampler(dat.Sampler):
    """
    A custom batchsampler
    """
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 adapt_dur=800,
                 adapt_token_num=150,
                 min_batch_size=4,
                 distributed=False):
        self.distributed = distributed
        if distributed:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        batches = self._work_batch_index(dataset,
                                         adapt_dur,
                                         adapt_token_num,
                                         batch_size,
                                         min_batch_size=min_batch_size)
        if distributed:
            self.num_batches = len(batches) // self.world_size
        else:
            self.num_batches = len(batches)
        self.batches = batches
        self.shuffle = shuffle
        self.epoch = 0

    def _work_batch_index(self,
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
            if self.shuffle:
                indices = th.randperm(self.num_batches).tolist()
            else:
                indices = th.arange(self.num_batches).tolist()
        self.epoch += 1
        for i in indices:
            yield list(range(*self.batches[i]))

    def __len__(self):
        return self.num_batches