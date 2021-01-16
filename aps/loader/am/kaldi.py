#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Dataloader for kaldi features
"""
import torch as th

from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Iterable, Optional
from kaldi_python_io import ScriptReader
from aps.loader.am.utils import AsrDataset, AsrDataLoader
from aps.libs import ApsRegisters
from aps.const import IGNORE_ID


@ApsRegisters.loader.register("am@kaldi")
def DataLoader(train: bool = True,
               distributed: bool = False,
               feats_scp: str = "",
               text: str = "",
               utt2dur: str = "",
               vocab_dict: Optional[Dict] = None,
               max_dur: float = 3000,
               min_dur: float = 40,
               adapt_dur: float = 800,
               min_token_num: int = 1,
               max_token_num: int = 400,
               adapt_token_num: int = 150,
               skip_utts: str = "",
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               max_batch_size: int = 32,
               min_batch_size: int = 4) -> Iterable[Dict]:
    """
    Args:
        train: in training mode or not
        distributed: in distributed mode or not
        feats_scp: path of the feature script
        text: path of the text/token file
        utt2dur: path of the duration file (should be utt2num_frames here)
        skip_utts: skips utterances if the key is in this file
        vocab_dict: vocabulary dictionary object
        {min|max}_dur: discard utterance when #num_frames not in [min_dur, max_dur]
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        adapt_dur|adapt_token_num: used in adaptive mode dataloader
        batch_mode: adaptive or constraint
        num_workers: number of the workers
        max_batch_size: maximum #batch_size
        min_batch_size: minimum #batch_size
    """
    dataset = Dataset(feats_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      skip_utts=skip_utts,
                      min_token_num=min_token_num,
                      max_token_num=max_token_num,
                      max_frame_num=max_dur,
                      min_frame_num=min_dur)
    return AsrDataLoader(dataset,
                         egs_collate,
                         shuffle=train,
                         distributed=distributed,
                         num_workers=num_workers,
                         adapt_dur=adapt_dur,
                         adapt_token_num=adapt_token_num,
                         batch_mode=batch_mode,
                         max_batch_size=max_batch_size,
                         min_batch_size=min_batch_size)


class Dataset(AsrDataset):
    """
    Dataset for kaldi features
    Args:
        feats_scp: path of the feature script
        text: path of the text/token file
        utt2dur: path of the duration file (should be utt2num_frames here)
        vocab_dict: vocabulary dictionary object
        skip_utts: skips utterances if the key is in this file
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_frame_num: discard utterance when #num_frames not in [#min_frame_num, #max_frame_num]
    """

    def __init__(self,
                 feats_scp: str,
                 text: str,
                 utt2num_frames: str,
                 vocab_dict: Optional[Dict],
                 skip_utts: str = "",
                 min_token_num: int = 1,
                 max_token_num: int = 400,
                 max_frame_num: float = 3000,
                 min_frame_num: float = 40) -> None:
        feats_reader = ScriptReader(feats_scp)
        super(Dataset, self).__init__(feats_reader,
                                      text,
                                      utt2num_frames,
                                      vocab_dict,
                                      max_dur=max_frame_num,
                                      min_dur=min_frame_num,
                                      dur_axis=0,
                                      skip_utts=skip_utts,
                                      min_token_num=min_token_num,
                                      max_token_num=max_token_num)


def egs_collate(egs: Dict) -> Dict:
    """
    Batch collate function, return with dict object with keys:
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
