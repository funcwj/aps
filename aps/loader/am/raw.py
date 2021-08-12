#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Dataloader for raw waveforms in asr tasks
"""
import torch as th

from torch.nn.utils.rnn import pad_sequence
from typing import Dict, Iterable, Optional
from aps.loader.am.utils import ASRDataset, ASRDataLoader
from aps.loader.audio import AudioReader
from aps.const import IGNORE_ID
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("am@raw")
def DataLoader(train: bool = True,
               distributed: bool = False,
               wav_scp: str = "",
               sr: int = 16000,
               channel: int = -1,
               text: str = "",
               utt2dur: str = "",
               vocab_dict: Optional[Dict] = None,
               min_token_num: int = 1,
               max_token_num: int = 400,
               max_dur: float = 30,
               min_dur: float = 0.4,
               adapt_dur: float = 8,
               adapt_token_num: int = 150,
               skip_utts: str = "",
               batch_mode: str = "adaptive",
               num_workers: int = 0,
               max_batch_size: int = 32,
               min_batch_size: int = 4) -> Iterable[Dict]:
    """
    Return the raw waveform dataloader (for AM training)
    Args:
        train: in training mode or not
        distributed: in distributed mode or not
        sr: sample rate of the audio
        channel: which channel to load, -1 means all
        wav_scp: path of the audio script
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: dictionary object
        skip_utts: skips utterances that the file shows
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_dur: discard utterance when #num_frames is not in [#min_dur, #max_dur]
        adapt_dur|adapt_token_num: used in adaptive mode
        batch_mode: adaptive or constraint
        num_workers: number of the workers
        max_batch_size: maximum #batch_size
        min_batch_size: minimum #batch_size
    """
    dataset = Dataset(wav_scp,
                      text,
                      utt2dur,
                      vocab_dict,
                      sr=sr,
                      channel=channel,
                      skip_utts=skip_utts,
                      min_token_num=min_token_num,
                      max_token_num=max_token_num,
                      max_wav_dur=max_dur,
                      min_wav_dur=min_dur)
    return ASRDataLoader(dataset,
                         egs_collate,
                         shuffle=train,
                         distributed=distributed,
                         num_workers=num_workers,
                         adapt_dur=adapt_dur,
                         adapt_token_num=adapt_token_num,
                         batch_mode=batch_mode,
                         max_batch_size=max_batch_size,
                         min_batch_size=min_batch_size)


class Dataset(ASRDataset):
    """
    Dataset for raw waveform input
    Args:
        wav_scp: path of the audio script
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary object
        sr: sample rate of the audio
        channel: which channel to load, -1 means all
        skip_utts: skips utterances that the file shows
        audio_norm: loading normalized samples (-1, 1) when reading audio
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_wav_dur: discard utterance when duration is not in [min_wav_dur, max_wav_dur]
        adapt_wav_dur|adapt_token_num: used in adaptive mode
    """

    def __init__(self,
                 wav_scp: str,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 sr: int = 16000,
                 channel: int = -1,
                 skip_utts: str = "",
                 audio_norm: bool = True,
                 min_token_num: int = 1,
                 max_token_num: int = 400,
                 max_wav_dur: float = 30,
                 min_wav_dur: float = 0.4,
                 adapt_wav_dur: float = 8,
                 adapt_token_num: int = 150) -> None:
        audio_reader = AudioReader(wav_scp,
                                   sr=sr,
                                   channel=channel,
                                   norm=audio_norm)
        super(Dataset, self).__init__(audio_reader,
                                      text,
                                      utt2dur,
                                      vocab_dict,
                                      max_dur=max_wav_dur,
                                      min_dur=min_wav_dur,
                                      dur_axis=0,
                                      skip_utts=skip_utts,
                                      min_token_num=min_token_num,
                                      max_token_num=max_token_num)


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
        assert peek_dim in [1, 2]
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
