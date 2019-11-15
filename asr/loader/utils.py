#!/usr/bin/env python

# wujian@2019

import subprocess
import torch as th
import torch.utils.data as dat

from kaldi_python_io import Reader as BaseReader


def process_token(token, utt2dur, max_token_num=400, max_dur=3000, min_dur=40):
    utt2dur = BaseReader(utt2dur, value_processor=float)
    token_reader = BaseReader(token,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1)
    token_set = []
    for key, token in token_reader:
        L = len(token)
        if L > max_token_num:
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
    if len(token_set) < 10:
        raise RuntimeError("Less utterances, check data configurations")
    return token_set


def count_token(token_scp, vocab_size):
    """
    Count number of each token
    """
    token_count = th.zeros(vocab_size + 1, dtype=th.float32)
    token_reader = BaseReader(token_scp,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1)
    num_utts = 0
    for key, token in token_reader:
        if len(token):
            token_count[token] += 1
            num_utts += 1
        else:
            print(f"Empty utterance: {key}")
    token_count = th.clamp(token_count, min=1)
    token_count[-1] = num_utts
    return token_count


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
        if p.returncode is not 0:
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
                 min_batch_size=4):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batches = self._work_batch_index(adapt_dur,
                                              adapt_token_num,
                                              batch_size,
                                              min_batch_size=min_batch_size)
        self.num_batches = len(self.batches)

    def _work_batch_index(self,
                          adapt_dur,
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
            factor = max(cur_ilen // adapt_dur, cur_olen // adapt_token_num)
            cur_bz = int(max(min_batch_size, batch_size // (1 + factor)))
            idx_bz.append((beg, beg + cur_bz))
            beg += cur_bz
        return idx_bz

    def __iter__(self):
        order = th.randperm(self.num_batches) if self.shuffle else th.arange(
            0, self.num_batches - 1, dtype=th.int32)
        for i in order.tolist():
            beg, end = self.batches[i]
            yield list(range(beg, end))

    def __len__(self):
        return self.num_batches