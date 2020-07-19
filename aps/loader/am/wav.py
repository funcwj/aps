#!/usr/bin/env python

# wujian@2019
"""
Dataloader for raw waveforms in asr tasks
"""
import io

import numpy as np
import torch as th
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence

from aps.loader.am.utils import process_token, BatchSampler
from aps.loader.audio import WaveReader


def DataLoader(train=True,
               distributed=False,
               wav_scp="",
               sr=16000,
               channel=-1,
               token="",
               utt2dur="",
               max_token_num=400,
               max_dur=30,
               min_dur=0.4,
               adapt_dur=8,
               adapt_token_num=150,
               batch_size=32,
               num_workers=0,
               min_batch_size=4):
    dataset = Dataset(wav_scp,
                      token,
                      utt2dur,
                      sr=sr,
                      channel=channel,
                      max_token_num=max_token_num,
                      max_wav_dur=max_dur,
                      min_wav_dur=min_dur)
    return WaveDataLoader(dataset,
                          shuffle=train,
                          distributed=distributed,
                          num_workers=num_workers,
                          adapt_wav_dur=adapt_dur,
                          adapt_token_num=adapt_token_num,
                          batch_size=batch_size,
                          min_batch_size=min_batch_size)


class Dataset(dat.Dataset):
    """
    Dataset for raw waveform
    """
    def __init__(self,
                 wav_scp="",
                 token="",
                 utt2dur="",
                 sr=16000,
                 channel=-1,
                 max_token_num=400,
                 max_wav_dur=30,
                 min_wav_dur=0.4,
                 adapt_wav_dur=8,
                 adapt_token_num=150):
        self.wav_reader = WaveReader(wav_scp,
                                     sr=sr,
                                     channel=channel,
                                     norm=True)
        self.token_reader = process_token(token,
                                          utt2dur,
                                          max_token_num=max_token_num,
                                          max_dur=max_wav_dur,
                                          min_dur=min_wav_dur)

    def __getitem__(self, idx):
        tok = self.token_reader[idx]
        key = tok["key"]
        wav = self.wav_reader[key]
        return {
            "dur": wav.shape[-1],
            "len": tok["len"],
            "wav": wav,
            "tok": tok["tok"]
        }

    def __len__(self):
        return len(self.token_reader)


def egs_collate(egs):
    def pad_seq(seq, value=0):
        peek_dim = seq[0].dim()
        if peek_dim not in [1, 2]:
            raise RuntimeError(
                "Now only supporting pad_sequence for 1/2D tensor")
        if peek_dim == 2:
            # C x S => S x C
            seq = [s.transpose(0, 1) for s in seq]
        # N x S x C
        pad_mat = pad_sequence(seq, batch_first=True, padding_value=value)
        if peek_dim == 2:
            pad_mat = pad_mat.transpose(1, 2)
        return pad_mat

    return {
        "src_pad":  # N x S or N x C x S
        pad_seq([th.from_numpy(eg["wav"]) for eg in egs], value=0),
        "tgt_pad":  # N x T
        pad_seq([th.as_tensor(eg["tok"]) for eg in egs], value=-1),
        "src_len":  # N, number of the frames
        th.tensor([eg["dur"] for eg in egs], dtype=th.int64),
        "tgt_len":  # N, length of the tokens
        th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }


class WaveDataLoader(object):
    """
    Acoustic dataloader for seq2seq model training
    """
    def __init__(self,
                 dataset,
                 shuffle=True,
                 distributed=False,
                 num_workers=0,
                 adapt_wav_dur=8,
                 adapt_token_num=150,
                 batch_size=32,
                 min_batch_size=4):
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
                               distributed=distributed,
                               adapt_dur=adapt_wav_dur,
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