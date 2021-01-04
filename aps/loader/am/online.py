#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Online simulation dataloader for ASR
"""
import numpy as np

from typing import Optional, Dict, Iterable
from aps.loader.am.utils import AsrDataset, AsrDataLoader
from aps.loader.se.online import SimuOptionsDataset
from aps.loader.am.raw import egs_collate
from aps.const import MAX_INT16
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("am_online")
def DataLoader(train: bool = True,
               distributed: bool = False,
               simu_cfg: str = "",
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
    Return the online simulation dataloader (for AM training)
    Args:
        train: in training mode or not
        distributed: in distributed mode or not
        simu_cfg: path of the audio simulation configuration
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: dictionary object
        audio_norm: loading normalized samples (-1, 1) when reading audio
        max_token_num: filter the utterances if the token number exceeds #max_token_num
        min_dur|max_dur: discard utterance when #num_frames is not in [min_dur, max_dur]
        adapt_dur|adapt_token_num: used in adaptive mode
        skip_utts: skips utterances that the file shows
        batch_size: maximum #batch_size
        batch_mode: adaptive or constraint
        num_workers: number of the workers
        min_batch_size: minimum #batch_size
    """
    dataset = Dataset(simu_cfg,
                      text,
                      utt2dur,
                      vocab_dict,
                      audio_norm=audio_norm,
                      skip_utts=skip_utts,
                      max_token_num=max_token_num,
                      max_wav_dur=max_dur,
                      min_wav_dur=min_dur)
    return AsrDataLoader(dataset,
                         egs_collate,
                         shuffle=train,
                         distributed=distributed,
                         num_workers=num_workers,
                         adapt_dur=adapt_dur,
                         adapt_token_num=adapt_token_num,
                         batch_size=batch_size,
                         batch_mode=batch_mode,
                         min_batch_size=min_batch_size)


class AsrSimuReader(SimuOptionsDataset):
    """
    Simulation audio reader for ASR task
    Args:
        simu_cfg: path of the audio simulation configuraton file
    """

    def __init__(self, simu_cfg: str, audio_norm: bool = True) -> None:
        super(AsrSimuReader, self).__init__(simu_cfg, return_in_egs=["mix"])
        self.audio_norm = audio_norm

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Args:
            index: index ID
        Return:
            egs: simulated audio
        """
        opts_str = self.simu_cfg[index]
        simu = self._simu(opts_str)["mix"]
        if self.audio_norm:
            simu = np.around(simu * MAX_INT16)
        return simu


class Dataset(AsrDataset):
    """
    Dataset for raw waveform input
    Args:
        simu_cfg: path of the simulation configuration file
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary object
        audio_norm: loading normalized samples (-1, 1) when reading audio
        skip_utts: skips utterances that the file shows
        max_token_num: filter the utterances if the token number exceeds #max_token_num
        {min|max}_wav_dur: discard utterance when duration is not in [min_wav_dur, max_wav_dur]
        adapt_wav_dur|adapt_token_num: used in adaptive mode
    """

    def __init__(self,
                 simu_cfg: str,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 audio_norm: bool = True,
                 skip_utts: str = "",
                 max_token_num: int = 400,
                 max_wav_dur: float = 30,
                 min_wav_dur: float = 0.4,
                 adapt_wav_dur: float = 8,
                 adapt_token_num: int = 150) -> None:
        audio_reader = AsrSimuReader(simu_cfg, audio_norm=audio_norm)
        super(Dataset, self).__init__(audio_reader,
                                      text,
                                      utt2dur,
                                      vocab_dict,
                                      skip_utts=skip_utts,
                                      max_dur=max_wav_dur,
                                      min_dur=min_wav_dur,
                                      max_token_num=max_token_num,
                                      duration_axis=-1)
