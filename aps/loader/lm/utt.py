#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""
for RNNLM (utterance corpus)
"""
import warnings
import torch as th

import torch.utils.data as dat

from torch.nn.utils.rnn import pad_sequence
from typing import NoReturn, List, Dict, Tuple, Optional, Callable, Iterator, Iterable
from kaldi_python_io import Reader as BaseReader
from aps.libs import ApsRegisters


@ApsRegisters.loader.register("lm_utt")
def DataLoader(text: str = "",
               vocab_dict: Optional[Dict] = None,
               train: bool = True,
               sos: int = -1,
               eos: int = -1,
               faster: bool = False,
               min_token_num: int = 2,
               max_token_num: int = 2000,
               adapt_token_num: int = 400,
               min_batch_size: int = 8,
               batch_size: int = 64,
               num_workers: int = 0) -> Iterable[Dict]:
    dataset = Dataset(text,
                      vocab_dict,
                      faster=faster,
                      min_token_num=min_token_num,
                      max_token_num=max_token_num)
    return UttDataLoader(dataset,
                         sos=sos,
                         eos=eos,
                         shuffle=train,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         min_batch_size=min_batch_size,
                         adapt_token_num=adapt_token_num)


class BatchSampler(dat.Sampler):
    """
    A custom batchsampler
    """

    def __init__(self,
                 token_set: List[List[int]],
                 batch_size: int,
                 min_batch_size: int = 8,
                 adapt_token_num: int = 400,
                 shuffle: bool = False) -> None:
        self.min_batch_size = min_batch_size
        self.adapt_token_num = adapt_token_num
        self.genfunc = th.randperm if shuffle else th.arange
        self.batches = []
        chunk_size = 10000 * batch_size
        for i in range(0, len(token_set), chunk_size):
            indices = self._sort_indices(token_set[i:i + chunk_size],
                                         batch_size)
            self.batches += indices

    def _sort_indices(self, subset: List[int],
                      batch_size: int) -> List[List[int]]:
        utts_len = [len(seq) for seq in subset]
        # short -> long
        desc_idx = th.argsort(th.tensor(utts_len, dtype=th.int32),
                              descending=False).tolist()
        batches = []
        beg, cur_bz = 0, batch_size
        while beg + cur_bz <= len(desc_idx):
            cur_len = utts_len[desc_idx[beg]]
            factor = cur_len // self.adapt_token_num
            cur_bz = int(max(self.min_batch_size, batch_size // (1 + factor)))
            batches.append(desc_idx[beg:beg + cur_bz])
            beg += cur_bz
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        indices = self.genfunc(len(self.batches)).tolist()
        for i in indices:
            yield self.batches[i]

    def __len__(self) -> int:
        return len(self.batches)


def parse_faster(scp_path: str,
                 value_processor: Callable = lambda x: x,
                 num_tokens: int = 2,
                 restrict: bool = True) -> Dict:
    scp_dict = dict()
    with open(scp_path, "r") as f:
        raw_lines = f.readlines()
        for idx, raw_line in enumerate(raw_lines):
            scp_tokens = raw_line.strip().split()
            if scp_tokens[-1] == "|":
                key, value = scp_tokens[0], " ".join(scp_tokens[1:])
            else:
                token_len = len(scp_tokens)
                if num_tokens >= 2 and token_len != num_tokens or restrict and token_len < 2:
                    raise RuntimeError(f"For {scp_path}, format error " +
                                       f"in line[{idx:d}]: {raw_line}")
                if num_tokens == 2:
                    key, value = scp_tokens
                else:
                    key, value = scp_tokens[0], scp_tokens[1:]
            if key in scp_dict:
                raise ValueError(
                    f"Duplicate key \'{key}\' exists in {scp_path}")
            scp_dict[key] = value_processor(value)
    return scp_dict


class FasterTextReader(object):
    """
    To make it faster to load large text files
    """

    def __init__(self,
                 scp_path: str,
                 value_processor: Callable = lambda x: x,
                 num_tokens: int = 2,
                 restrict: bool = True) -> None:
        self.index_dict = parse_faster(scp_path,
                                       value_processor=value_processor,
                                       num_tokens=num_tokens,
                                       restrict=restrict)
        self.index_keys = list(self.index_dict.keys())

    def _load(self, key: str) -> List[str]:
        return self.index_dict[key]

    def __len__(self) -> int:
        return len(self.index_dict)

    def __contains__(self, key: str) -> bool:
        return key in self.index_dict

    def __iter__(self) -> Iterator[Tuple[str, List[str]]]:
        for key in self.index_keys:
            yield key, self._load(key)


class Dataset(dat.Dataset):
    """
    Dataset for token corpus
    """

    def __init__(self,
                 text: str,
                 vocab_dict: Optional[Dict],
                 faster: bool = False,
                 min_token_num: int = 2,
                 max_token_num: int = 2000,
                 eos: Optional[int] = None) -> None:
        reader_cls = FasterTextReader if faster else BaseReader
        if vocab_dict:
            text_reader = reader_cls(text, num_tokens=-1, restrict=False)
        else:
            text_reader = reader_cls(
                text,
                value_processor=lambda tok: list(map(int, tok)),
                num_tokens=-1,
                restrict=False)
        self.token_set = []
        for key, tokens in text_reader:
            if len(tokens) < min_token_num:
                warnings.warn(f"Pass utterance that is too short: {key}")
            elif len(tokens) > max_token_num:
                warnings.warn(f"Pass utterance that is too long: {key}")
            else:
                if vocab_dict:
                    toks = []
                    for t in tokens:
                        toks.append(vocab_dict[t] if t in
                                    vocab_dict else vocab_dict["<unk>"])
                else:
                    toks = tokens
                if eos is None:
                    self.token_set.append(toks)
                else:
                    self.token_set.append(toks + [eos])

    def __getitem__(self, index: int) -> List[int]:
        return self.token_set[index]

    def __len__(self) -> int:
        return len(self.token_set)


class UttDataLoader(dat.DataLoader):
    """
    DataLoader for LM training
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 sos: int = -1,
                 eos: int = -1,
                 shuffle: bool = True,
                 batch_size: int = 64,
                 num_workers: int = 0,
                 adapt_token_num: int = 400,
                 min_batch_size: int = 8) -> None:
        if sos < 0 or eos < 0:
            raise ValueError(f"Invalid sos/eos value: {sos}/{eos}")
        self.eos = eos
        self.sos = sos
        sampler = BatchSampler(dataset.token_set,
                               batch_size,
                               shuffle=shuffle,
                               min_batch_size=min_batch_size,
                               adapt_token_num=adapt_token_num)
        super(UttDataLoader, self).__init__(dataset,
                                            batch_sampler=sampler,
                                            num_workers=num_workers,
                                            collate_fn=self.egs_collate)

    def egs_collate(self, egs):
        ntok = [len(eg) + 1 for eg in egs]
        sos_egs = [th.as_tensor([self.sos] + eg) for eg in egs]
        egs_eos = [th.as_tensor(eg + [self.eos]) for eg in egs]
        return {
            "src": pad_sequence(sos_egs, batch_first=True, padding_value=0),
            "tgt": pad_sequence(egs_eos, batch_first=True, padding_value=-1),
            "len": th.tensor(ntok, dtype=th.int64)
        }

    def set_epoch(self, epoch: int) -> NoReturn:
        pass
