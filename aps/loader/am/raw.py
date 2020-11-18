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
from aps.loader.am.utils import process_token, BatchSampler
from aps.loader.audio import AudioReader
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
               adapt_dur: float = 8,
               adapt_token_num: int = 150,
               batch_size: int = 32,
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               min_batch_size: int = 4) -> Iterable[Dict]:
    dataset = Dataset(wav_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      sr=sr,
                      channel=channel,
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


class Dataset(dat.Dataset):
    """
    Dataset for raw waveform
    """

    def __init__(self,
                 wav_scp: str,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 sr: int = 16000,
                 channel: int = -1,
                 max_token_num: int = 400,
                 max_wav_dur: float = 30,
                 min_wav_dur: float = 0.4,
                 adapt_wav_dur: float = 8,
                 adapt_token_num: int = 150) -> None:
        self.audio_reader = AudioReader(wav_scp,
                                        sr=sr,
                                        channel=channel,
                                        norm=True)
        self.token_reader = process_token(text,
                                          utt2dur,
                                          vocab_dict,
                                          max_token_num=max_token_num,
                                          max_dur=max_wav_dur,
                                          min_dur=min_wav_dur)

    def __getitem__(self, idx: int) -> Dict:
        tok = self.token_reader[idx]
        key = tok["key"]
        wav = self.audio_reader[key]
        if wav is None:
            return {"dur": None, "len": None, "wav": wav, "tok": None}
        else:
            return {
                "dur": wav.shape[-1],
                "len": tok["len"],
                "wav": wav,
                "tok": tok["tok"]
            }

    def __len__(self) -> int:
        return len(self.token_reader)


def egs_collate(egs: Dict) -> Dict:

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

    egs = {
        # N x S or N x C x S
        "src_pad":
            pad_seq([
                th.from_numpy(eg["wav"]) for eg in egs if eg["wav"] is not None
            ],
                    value=0),
        # N x T
        "tgt_pad":
            pad_seq([
                th.as_tensor(eg["tok"]) for eg in egs if eg["tok"] is not None
            ],
                    value=-1),
        # N, number of the frames
        "src_len":
            th.tensor([eg["dur"] for eg in egs if eg["dur"] is not None],
                      dtype=th.int64),
        # N, length of the tokens
        "tgt_len":
            th.tensor([eg["len"] for eg in egs if eg["len"] is not None],
                      dtype=th.int64)
    }
    return egs


class AudioDataLoader(dat.DataLoader):
    """
    Acoustic dataloader for seq2seq model training
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
                                              num_workers=num_workers,
                                              collate_fn=egs_collate)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
