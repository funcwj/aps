#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
Online simulation dataloader for ASR
"""
import numpy as np

from typing import Optional, Dict, Iterable
from aps.loader.am.utils import CommonASRDataset, CommonASRDataLoader
from aps.loader.se.simu_cmd import CommandOptionsDataset
from aps.loader.am.raw import egs_collate
from aps.libs import ApsRegisters


class SimuCmdReader(CommandOptionsDataset):
    """
    Simulation audio reader for ASR task
    Args:
        simu_cfg: path of the audio simulation configuraton file
    """

    def __init__(self, simu_cfg: str) -> None:
        super(SimuCmdReader, self).__init__(simu_cfg, return_in_egs=["mix"])

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Args:
            index: index ID
        Return:
            egs: simulated audio
        """
        opts_str = self.simu_cfg[index]
        return self._simu(opts_str)["mix"]


@ApsRegisters.loader.register("am@simu_cmd")
def DataLoader(train: bool = True,
               distributed: bool = False,
               simu_cfg: str = "",
               text: str = "",
               utt2dur: str = "",
               vocab_dict: Optional[Dict] = None,
               tokenizer: str = "",
               tokenizer_kwargs: Dict = {},
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
    Return the online simulation dataloader (for AM training, command version)
    Args:
        train: in training mode or not
        distributed: in distributed mode or not
        simu_cfg: path of the audio simulation configuration
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary
        tokenizer: tokenizer name (for on-the-fly tokenizer)
        tokenizer_kwargs: argument options for tokenizer
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_dur: discard utterance when audio length is not in [#min_dur, #max_dur]
        adapt_dur|adapt_token_num: used in adaptive mode
        skip_utts: skips utterances that the file shows
        batch_mode: adaptive or constraint
        num_workers: number of the workers
        max_batch_size: maximum #batch_size
        min_batch_size: minimum #batch_size
    """
    audio_reader = SimuCmdReader(simu_cfg)
    dataset = CommonASRDataset(audio_reader,
                               text,
                               utt2dur,
                               vocab_dict,
                               tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs,
                               max_dur=max_dur,
                               min_dur=min_dur,
                               dur_axis=-1,
                               skip_utts=skip_utts,
                               min_token_num=min_token_num,
                               max_token_num=max_token_num)
    return CommonASRDataLoader(dataset,
                               egs_collate,
                               shuffle=train,
                               distributed=distributed,
                               num_workers=num_workers,
                               adapt_dur=adapt_dur,
                               adapt_token_num=adapt_token_num,
                               batch_mode=batch_mode,
                               max_batch_size=max_batch_size,
                               min_batch_size=min_batch_size)
