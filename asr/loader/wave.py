#!/usr/bin/env python

# wujian@2019
"""
Dataloader for raw waveforms in asr tasks
"""
import numpy as np
import torch as th
import soundfile as sf
import torch.utils.data as dat

from io import BytesIO

from torch.nn.utils.rnn import pad_sequence
from kaldi_python_io import Reader as BaseReader

from .utils import process_token, run_command, BatchSampler

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


def wave_loader(train=True,
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
    return DataLoader(dataset,
                      shuffle=train,
                      distributed=distributed,
                      num_workers=num_workers,
                      adapt_wav_dur=adapt_dur,
                      adapt_token_num=adapt_token_num,
                      batch_size=batch_size,
                      min_batch_size=min_batch_size)


def read_wav(fname,
             beg=0,
             end=None,
             norm=True,
             transpose=True,
             return_sr=False):
    """
    Read wave files using soundfile (support multi-channel)
    args:
        fname: file name or object
        beg, end: begin and end index for chunk-level reading
        norm: normalized samples between -1 and 1
        return_sr: return audio sample rate
    return:
        samps: in shape C x N
        sr: sample rate
    """
    # samps: N x C or N
    #   N: number of samples
    #   C: number of channels
    samps, sr = sf.read(fname,
                        start=beg,
                        stop=end,
                        dtype="float32" if norm else "int16")
    if not norm:
        samps = samps.astype("float32")
    # put channel axis first
    # N x C => C x N
    if samps.ndim != 1 and transpose:
        samps = np.transpose(samps)
    if return_sr:
        return sr, samps
    return samps


class WaveReader(BaseReader):
    """
        Sequential/Random Reader for single/multiple channel wave
        Format of wav.scp follows Kaldi's definition:
            key1 /path/to/key1.wav
            ...
        or
            key1 sox /home/data/key1.wav -t wav - remix 1 |
            ...
    """
    def __init__(self, wav_scp, sr=16000, norm=True, channel=-1):
        super(WaveReader, self).__init__(wav_scp, num_tokens=2)
        self.sr = sr
        self.ch = channel
        self.norm = norm
        self.mngr = {}

    def _load(self, key):
        fname = self.index_dict[key]
        # return C x N or N
        if ":" in fname:
            tokens = fname.split(":")
            if len(tokens) != 2:
                raise RuntimeError(f"Value format error: {fname}")
            fname, offset = tokens[0], int(tokens[1])
            # get ark object
            if fname not in self.mngr:
                self.mngr[fname] = open(fname, "rb")
            wav_ark = self.mngr[fname]
            # seek and read
            wav_ark.seek(offset)
            sr, samps = read_wav(wav_ark, norm=self.norm, return_sr=True)
        else:
            if fname[-1] == "|":
                shell, _ = run_command(fname[:-1], wait=True)
                fname = BytesIO(shell)
            sr, samps = read_wav(fname, norm=self.norm, return_sr=True)
        # if given sample rate, check it
        if sr != self.sr:
            raise RuntimeError(f"Sample rate mismatch: {sr:d} vs {self.sr:d}")
        # get one channel
        if self.ch >= 0 and samps.ndim == 2:
            samps = samps[self.ch]
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


class DataLoader(object):
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