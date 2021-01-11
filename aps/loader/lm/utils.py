#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th
import torch.utils.data as dat

from typing import List


def filter_utts(dataset: dat.Dataset,
                min_token_num: int = 4,
                max_token_num: int = 1000) -> List[int]:
    """
    Return utterance index used for training (pass short/long utterances)
    """
    kept_index = []
    filter_sutt, filter_lutt = 0, 0
    for index, tokseq in enumerate(dataset):
        tok_len = len(tokseq)
        if tok_len < min_token_num:
            filter_sutt += 1
        elif tok_len > max_token_num:
            filter_lutt += 1
        else:
            kept_index.append(index)
    if filter_lutt or filter_sutt:
        ratio_lutt = filter_lutt * 100.0 / len(dataset)
        ratio_sutt = filter_sutt * 100.0 / len(dataset)
        warnings.warn(
            f"filter {ratio_lutt:.2f}% long utterance & {ratio_sutt:.2f}% " +
            "short utterances...")
    return kept_index


def concat_data(batch_size: int,
                dataset: dat.Dataset,
                sampler: dat.Sampler,
                sos: int = 0,
                eos: int = 1) -> th.Tensor:
    """
    Concatenate data sequence in the dataset
    """
    data = []
    for index in sampler:
        data += ([sos] + dataset[index] + [eos])
    truncated = (len(data) // batch_size) * batch_size
    data = th.tensor(data, dtype=th.int64)
    batch = data[:truncated].view(batch_size, -1)
    return batch
