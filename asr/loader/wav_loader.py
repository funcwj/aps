#!/usr/bin/env python

# wujian@2019

import numpy as np
import torch as th
import soundfile as sf
import scipy.io.wavfile as wf
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from kaldi_python_io import Reader as BaseReader

from .utils import process_token, BatchSampler

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


def make_dataloader(train=True,
                    wav_scp="",
                    sr=16000,
                    token_scp="",
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
                      token_scp,
                      utt2dur,
                      sr=sr,
                      max_token_num=max_token_num,
                      max_wav_dur=max_dur,
                      min_wav_dur=min_dur)
    return DataLoader(dataset,
                      shuffle=train,
                      num_workers=num_workers,
                      adapt_wav_dur=adapt_dur,
                      adapt_token_num=adapt_token_num,
                      batch_size=batch_size,
                      min_batch_size=min_batch_size)


def read_wav(fname, beg=None, end=None, norm=True, return_sr=False):
    """
    Read wave files using scipy.io.wavfile (support multi-channel)
    """
    # samps_int16: N x C or N
    #   N: number of samples
    #   C: number of channels
    if beg is not None:
        samps_int16, sr = sf.read(fname, start=beg, stop=end, dtype="int16")
    else:
        sr, samps_int16 = wf.read(fname)
    # N x C => C x N
    samps = samps_int16.astype(np.float32)
    # put channel axis first
    if samps.ndim != 1:
        samps = np.transpose(samps)
    # normalize like MATLAB and librosa
    if norm:
        samps = samps / MAX_INT16
    if return_sr:
        return sr, samps
    return samps


class WaveReader(BaseReader):
    """
        Sequential/Random Reader for single/multiple channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/wav
            ...
    """
    def __init__(self, wav_scp, sr=16000, norm=True):
        super(WaveReader, self).__init__(wav_scp)
        self.sr = sr
        self.norm = norm

    def _load(self, key):
        # return C x N or N
        sr, samps = read_wav(self.index_dict[key],
                             norm=self.norm,
                             return_sr=True)
        # if given samp_rate, check it
        if self.sr is not None and sr != self.sr:
            raise RuntimeError("Sample rate mismatch: {:d} vs {:d}".format(
                sr, self.sr))
        return samps

    def nsamps(self, key):
        """
        Number of samples
        """
        data = self._load(key)
        return data.shape[-1]

    def power(self, key):
        """
        Power of utterance
        """
        data = self._load(key)
        s = data if data.ndim == 1 else data[0]
        return np.linalg.norm(s, 2)**2 / data.size

    def duration(self, key):
        """
        Utterance duration
        """
        N = self.nsamps(key)
        return N / self.sr


class Dataset(dat.Dataset):
    """
    Dataset for raw waveform
    """
    def __init__(self,
                 wav_scp="",
                 token_scp="",
                 utt2dur="",
                 sr=16000,
                 max_token_num=400,
                 max_wav_dur=30,
                 min_wav_dur=0.4,
                 adapt_wav_dur=8,
                 adapt_token_num=150):
        self.wav_reader = WaveReader(wav_scp, sr=sr)
        self.token_reader = process_token(token_scp,
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
            "token": tok["tok"]
        }

    def __len__(self):
        return len(self.token_reader)


def egs_collate(egs):
    def pad_seq(olist, value=0):
        return pad_sequence(olist, batch_first=True, padding_value=value)

    return {
        "x_pad":  # N x S
        pad_seq([th.from_numpy(eg["wav"]) for eg in egs], value=0),
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
                 adapt_wav_dur=8,
                 adapt_token_num=150,
                 batch_size=32,
                 min_batch_size=4):
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
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