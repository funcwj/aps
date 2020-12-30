#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Dataloader for raw waveforms in asr tasks
"""

import torch as th
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence

from typing import Dict, Iterable, Optional, NoReturn
from aps.loader.am.utils import AsrDataset, TokenReader, BatchSampler
from aps.loader.audio import AudioReader
from aps.const import IGNORE_ID
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("am_raw")
def DataLoader(train: bool = True,
               distributed: bool = False,
               wav_scp: str = "",
               sr: int = 16000,
               channel: int = -1,
               text: str = "",
               utt2dur: str = "",
               vocab_dict: Optional[Dict] = None,
               max_token_num: int = 400,
               max_dur: float = 30,
               min_dur: float = 0.4,
               audio_norm: bool = True,
               adapt_dur: float = 8,
               adapt_token_num: int = 150,
               skip_utts: str = "",
               batch_size: int = 32,
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               min_batch_size: int = 4) -> Iterable[Dict]:
    """
    Return the raw waveform dataloader (for AM training)
    Args:
        train: in training mode or not
        distributed: in distributed mode or not
        sr: sample rate of the audio
        channel: which channel to load, -1 means all
        audio_norm: loading normalized samples (-1, 1) when reading audio
        wav_scp: path of the audio script
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: dictionary object
        min_dur|max_dur: discard utterance when #num_frames is not in [min_dur, max_dur]
        skip_utts: skips utterances that the file shows
        adapt_dur|adapt_token_num: used in adaptive mode
        batch_size: maximum #batch_size
        batch_mode: adaptive or constraint
        num_workers: number of the workers
        min_batch_size: minimum #batch_size
    """
    dataset = Dataset(wav_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      sr=sr,
                      channel=channel,
                      audio_norm=audio_norm,
                      skip_utts=skip_utts,
                      max_token_num=max_token_num,
                      max_wav_dur=max_dur,
                      min_wav_dur=min_dur)
    return AudioDataLoader(dataset,
                           shuffle=train,
                           distributed=distributed,
                           num_workers=num_workers,
                           adapt_wav_dur=adapt_dur,
                           adapt_token_num=adapt_token_num,
                           batch_size=batch_size,
                           batch_mode=batch_mode,
                           min_batch_size=min_batch_size)


class Dataset(AsrDataset):
    """
    Dataset for raw waveform input
    Args:
        wav_scp: path of the audio script
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary object
        sr: sample rate of the audio
        channel: which channel to load, -1 means all
        audio_norm: loading normalized samples (-1, 1) when reading audio
        {min|max}_wav_dur: discard utterance when duration is not in [min_wav_dur, max_wav_dur]
        skip_utts: skips utterances that the file shows
        adapt_wav_dur|adapt_token_num: used in adaptive mode
    """

    def __init__(self,
                 wav_scp: str,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 sr: int = 16000,
                 channel: int = -1,
                 audio_norm: bool = True,
                 skip_utts: str = "",
                 max_token_num: int = 400,
                 max_wav_dur: float = 30,
                 min_wav_dur: float = 0.4,
                 adapt_wav_dur: float = 8,
                 adapt_token_num: int = 150) -> None:
        audio_reader = AudioReader(wav_scp,
                                   sr=sr,
                                   channel=channel,
                                   norm=audio_norm)
        token_reader = TokenReader(text,
                                   utt2dur,
                                   vocab_dict,
                                   skip_utts=skip_utts,
                                   max_dur=max_wav_dur,
                                   min_dur=min_wav_dur,
                                   max_token_num=max_token_num)
        super(Dataset, self).__init__(audio_reader,
                                      token_reader,
                                      duration_axis=-1)


def egs_collate(egs: Dict) -> Dict:
    """
    Batch collate function, return dict object with keys:
        #utt: batch size, int
        #tok: token size, int
        src_pad: raw waveforms, N x (C) x S
        tgt_pad: N x T
        src_len: number of the frames, N
        tgt_len: length of the tokens, N
    """

    def pad_seq(seq, value=0):
        peek_dim = seq[0].dim()
        if peek_dim not in [1, 2]:
            raise RuntimeError("Now only supports 1/2D tensor")
        # C x S => S x C
        if peek_dim == 2:
            seq = [s.transpose(0, 1) for s in seq]
        # N x S x C
        pad_mat = pad_sequence(seq, batch_first=True, padding_value=value)
        # N x (C) x S
        if peek_dim == 2:
            pad_mat = pad_mat.transpose(1, 2)
        return pad_mat

    egs = {
        "#utt":
            len(egs),
        "#tok":  # add 1 as during training we pad sos
            sum([int(eg["len"]) + 1 for eg in egs]),
        "src_pad":
            pad_seq([th.from_numpy(eg["inp"]) for eg in egs], value=0),
        "tgt_pad":
            pad_seq([th.as_tensor(eg["ref"]) for eg in egs], value=IGNORE_ID),
        "src_len":
            th.tensor([eg["dur"] for eg in egs], dtype=th.int64),
        "tgt_len":
            th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }
    return egs


class AudioDataLoader(dat.DataLoader):
    """
    Raw waveform dataloader for E2E AM training
    Args:
        dataset: instance of dat.Dataset
        shuffle: shuffle mini-batches or not
        distributed: in distributed mode or not
        num_workers: number of the workers
        adapt_wav_dur|adapt_token_num: used in adaptive mode
        batch_size: maximum #batch_size
        batch_mode: adaptive or constraint
        min_batch_size: minimum #batch_size
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 shuffle: bool = True,
                 distributed: bool = False,
                 num_workers: int = 0,
                 adapt_wav_dur: float = 8,
                 adapt_token_num: int = 150,
                 batch_size: int = 32,
                 batch_mode: str = "adaptive",
                 min_batch_size: int = 4) -> None:
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
                               distributed=distributed,
                               batch_mode=batch_mode,
                               adapt_dur=adapt_wav_dur,
                               adapt_token_num=adapt_token_num,
                               min_batch_size=min_batch_size)

        super(AudioDataLoader, self).__init__(dataset,
                                              batch_sampler=sampler,
                                              collate_fn=egs_collate,
                                              num_workers=num_workers)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
