#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Dataloader for kaldi features
"""
import torch as th
import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Iterable, Optional, NoReturn
from kaldi_python_io import ScriptReader

from aps.loader.am.utils import process_token, BatchSampler
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("am_kaldi")
def DataLoader(train: bool = True,
               distributed: bool = False,
               feats_scp: str = "",
               text: str = "",
               utt2dur: str = "",
               vocab_dict: Optional[Dict] = None,
               max_token_num: int = 400,
               max_dur: float = 3000,
               min_dur: float = 40,
               adapt_dur: float = 800,
               adapt_token_num: int = 150,
               batch_size: int = 32,
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               min_batch_size: int = 4) -> Iterable[Dict]:
    dataset = Dataset(feats_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      max_token_num=max_token_num,
                      max_frame_num=max_dur,
                      min_frame_num=min_dur)
    return KaldiDataLoader(dataset,
                           shuffle=train,
                           distributed=distributed,
                           num_workers=num_workers,
                           adapt_frame_num=adapt_dur,
                           adapt_token_num=adapt_token_num,
                           batch_size=batch_size,
                           batch_mode=batch_mode,
                           min_batch_size=min_batch_size)


class Dataset(dat.Dataset):
    """
    Dataset for kaldi features
    """

    def __init__(self,
                 feats_scp: str,
                 text: str,
                 utt2num_frames: str,
                 vocab_dict: Optional[Dict],
                 max_token_num: int = 400,
                 max_frame_num: float = 3000,
                 min_frame_num: float = 40) -> None:
        self.feats_reader = ScriptReader(feats_scp)
        # sorted
        self.token_reader = process_token(text,
                                          utt2num_frames,
                                          vocab_dict,
                                          max_token_num=max_token_num,
                                          max_dur=max_frame_num,
                                          min_dur=min_frame_num)

    def __getitem__(self, idx: int) -> Dict:
        tok = self.token_reader[idx]
        key = tok["key"]
        return {
            "dur": tok["dur"],
            "len": tok["len"],
            "feats": self.feats_reader[key],
            "token": tok["tok"]
        }

    def __len__(self) -> int:
        return len(self.token_reader)


def egs_collate(egs: Dict) -> Dict:

    def pad_seq(olist, value=0):
        return pad_sequence(olist, batch_first=True, padding_value=value)

    return {
        # N x S
        "src_pad":
            pad_seq([th.from_numpy(eg["feats"].copy()) for eg in egs], value=0),
        # N x T
        "tgt_pad":
            pad_seq([th.as_tensor(eg["token"]) for eg in egs], value=-1),
        # N, number of the frames
        "src_len":
            th.tensor([int(eg["dur"]) for eg in egs], dtype=th.int64),
        # N, length of the tokens
        "tgt_len":
            th.tensor([eg["len"] for eg in egs], dtype=th.int64)
    }


class KaldiDataLoader(dat.DataLoader):
    """
    Acoustic dataloader for seq2seq model training
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 shuffle: bool = True,
                 distributed: bool = False,
                 num_workers: int = 0,
                 adapt_frame_num: float = 800,
                 adapt_token_num: int = 150,
                 batch_size: int = 32,
                 batch_mode: str = "adaptive",
                 min_batch_size: int = 4) -> None:
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
                               batch_mode=batch_mode,
                               distributed=distributed,
                               adapt_dur=adapt_frame_num,
                               adapt_token_num=adapt_token_num,
                               min_batch_size=min_batch_size)
        super(KaldiDataLoader, self).__init__(dataset,
                                              batch_sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=egs_collate)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
