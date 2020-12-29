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

from aps.loader.am.utils import AsrDataset, TokenReader, BatchSampler
from aps.libs import ApsRegisters
from aps.const import IGNORE_ID


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
               skip_utts: str = "",
               batch_size: int = 32,
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               min_batch_size: int = 4) -> Iterable[Dict]:
    dataset = Dataset(feats_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      skip_utts=skip_utts,
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


class Dataset(AsrDataset):
    """
    Dataset for kaldi features
    """

    def __init__(self,
                 feats_scp: str,
                 text: str,
                 utt2num_frames: str,
                 vocab_dict: Optional[Dict],
                 skip_utts: str = "",
                 max_token_num: int = 400,
                 max_frame_num: float = 3000,
                 min_frame_num: float = 40) -> None:
        feats_reader = ScriptReader(feats_scp)
        token_reader = TokenReader(text,
                                   utt2num_frames,
                                   vocab_dict,
                                   skip_utts=skip_utts,
                                   max_dur=max_frame_num,
                                   min_dur=min_frame_num,
                                   max_token_num=max_token_num)
        super(Dataset, self).__init__(feats_reader,
                                      token_reader,
                                      duration_axis=0)


def egs_collate(egs: Dict) -> Dict:
    """
    Batch collate, return with keys:
        #utt: batch size, int
        #tok: token size, int
        src_pad: kaldi features N x T x F
        tgt_pad: target tokens, N x T
        src_len: number of the frames, N
        tgt_len: length of the tokens, N
    """

    def pad_seq(olist, value=0):
        return pad_sequence(olist, batch_first=True, padding_value=value)

    return {
        "#utt":
            len(egs),
        "#tok":  # add 1 as during training we pad sos
            sum([int(eg["len"]) + 1 for eg in egs]),
        "src_pad":
            pad_seq([th.from_numpy(eg["inp"].copy()) for eg in egs], value=0),
        "tgt_pad":
            pad_seq([th.as_tensor(eg["ref"]) for eg in egs], value=IGNORE_ID),
        "src_len":
            th.tensor([int(eg["dur"]) for eg in egs], dtype=th.int64),
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
                                              collate_fn=egs_collate,
                                              num_workers=num_workers)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
