#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th

import torch.utils.data as dat
import aps.distributed as dist

from aps.tokenizer import Vocab
from typing import Dict, List, Tuple, NoReturn, Optional, Callable
from kaldi_python_io import Reader as BaseReader


def derive_indices(num_batches: int,
                   seed: int = 0,
                   shuffle: bool = True,
                   distributed: bool = False) -> List[int]:
    """
    Return indices for BatchSampler
    """
    if distributed:
        rank = dist.rank()
        world_size = dist.world_size()
        num_batches = num_batches * world_size
    if shuffle:
        g = th.Generator()
        g.manual_seed(seed)
        indices = th.randperm(num_batches, generator=g).tolist()
    else:
        indices = th.arange(num_batches).tolist()
    if distributed:
        return indices[rank:num_batches:world_size]
    else:
        return indices


class ASRDataset(dat.Dataset):
    """
    A base dataset class for AM training

    Args:
        input_reader: source feature reader instance
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary object
        dur_axis: duration axis index for input_reader
        skip_utts: skips utterances that the file shows
        {min|max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_dur: filter the utterances when length is not in [#min_wav_dur, #max_wav_dur]
    """

    def __init__(self,
                 input_reader,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 dur_axis: int = -1,
                 skip_utts: str = "",
                 max_token_num: int = 400,
                 min_token_num: int = 2,
                 max_dur: float = 3000,
                 min_dur: float = 40) -> None:
        self.input_reader = input_reader
        self.token_reader = TokenReader(text,
                                        utt2dur,
                                        vocab_dict,
                                        skip_utts=skip_utts,
                                        max_dur=max_dur,
                                        min_dur=min_dur,
                                        max_token_num=max_token_num,
                                        min_token_num=min_token_num)
        self.dur_axis = dur_axis

    def __getitem__(self, idx: int) -> Dict:
        tok = self.token_reader[idx]
        key = tok["key"]
        inp = self.input_reader[key]
        return {
            "dur": inp.shape[self.dur_axis],
            "inp": inp,
            "len": tok["len"],
            "ref": tok["tok"]
        }

    def __len__(self) -> int:
        return len(self.token_reader)


class TokenReader(object):
    """
    The token/text reader for ASR task. It will filter utterances that:
        1) length of the token not in [min_token_num, max_token_num]
        2) length of the audio not in [min_dur, max_dur]
        3) utterance's key is in skip_utts
    and tokenize reference files (from string tokens to int sequences)
    """

    def __init__(self,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 max_token_num: int = 400,
                 min_token_num: int = 2,
                 max_dur: float = 3000,
                 min_dur: float = 40,
                 skip_utts: str = ""):
        self.vocab_dict = Vocab(vocab_dict) if vocab_dict else None
        self.token_list = self._pre_process(text,
                                            utt2dur,
                                            max_dur=max_dur,
                                            min_dur=min_dur,
                                            skip_utts=skip_utts,
                                            max_token_num=max_token_num,
                                            min_token_num=min_token_num)
        if len(self.token_list) < 10:
            raise RuntimeError(
                f"Too less utterances: {len(self.token_list)}, " +
                "please check data configurations")

    def _pre_process(self,
                     text: str,
                     utt2dur: str,
                     max_token_num: int = 400,
                     min_token_num: int = 2,
                     skip_utts: str = "",
                     max_dur: float = 3000,
                     min_dur: float = 40) -> List[Dict]:
        """
        Preprocess function to filter the utterances
        """
        if skip_utts:
            with open(skip_utts, "r") as skip_fd:
                skip_keys = [k.strip() for k in skip_fd.readlines()]
        else:
            skip_keys = []
        utt2dur = BaseReader(utt2dur, value_processor=float)
        if self.vocab_dict:
            text_reader = BaseReader(text, num_tokens=-1, restrict=False)
        else:
            text_reader = BaseReader(
                text,
                value_processor=lambda tok: list(map(int, tok)),
                num_tokens=-1,
                restrict=False)
        token_set = []
        drop_utts = 0
        for key, tokens in text_reader:
            num_toks = len(tokens)
            if num_toks > max_token_num or num_toks < min_token_num:
                drop_utts += 1
                continue
            if key not in utt2dur:
                drop_utts += 1
                continue
            if key in skip_keys:
                continue
            num_frames = utt2dur[key]
            if num_frames < min_dur or num_frames > max_dur:
                drop_utts += 1
                continue
            token_set.append({
                "key": key,
                "dur": num_frames,
                "len": num_toks,
                "tok": tokens
            })
        # long -> short
        token_set = sorted(token_set, key=lambda d: d["dur"], reverse=True)
        if drop_utts:
            warnings.warn(f"Drop {drop_utts} utterances")
        return token_set

    def __getitem__(self, index):
        stats = self.token_list[index]
        # if processed, skip
        if self.vocab_dict and "vis" not in stats:
            # map from str sequences to int sequences
            stats["tok"] = self.vocab_dict(stats["tok"])
            stats["vis"] = True
        return stats

    def __len__(self) -> int:
        return len(self.token_list)


class BatchSampler(dat.Sampler):
    """
    A custom batch sampler that can used in distributed/non-distributed mode
    Args:
        dataset: dataset object
        max_batch_size: maximum #batch_size
        min_batch_size: minimum #batch_size
        shuffle: shuffle batches or not
        batch_mode: "adaptive" or "constraint"
        adapt_dur|adapt_token_num: used in adaptive mode, see _work_adapt_batch_index
        distributed: distributed or not
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 max_batch_size: int,
                 shuffle: bool = False,
                 batch_mode: str = "adaptive",
                 adapt_dur: float = 800,
                 adapt_token_num: int = 150,
                 min_batch_size: int = 4,
                 distributed: bool = False) -> None:
        if batch_mode not in ["adaptive", "constraint"]:
            raise ValueError(f"Unsupported batch mode: {batch_mode}")
        if batch_mode == "adaptive":
            batches = self._work_adapt_batch_index(dataset, adapt_dur,
                                                   adapt_token_num,
                                                   max_batch_size,
                                                   min_batch_size)
        else:
            batches = self._work_const_batch_index(dataset, max_batch_size)
        self.epoch = 0
        self.batches = batches
        self.shuffle = shuffle
        self.world_size = dist.world_size() if distributed else 1
        self.distributed = distributed
        self.num_batches = len(batches) // self.world_size

    def _work_const_batch_index(self, dataset: dat.Dataset,
                                max_batch_size: int) -> List[Tuple[int, int]]:
        """
        In constraint mode, the batch [utt_1, utt_2, ..., utt_N] satisfies
            sum([len(utt_1), ..., len(utt_N)]) <= #batch_size
        """
        beg = 0
        tot = len(dataset)
        cur_dur = 0
        idx_bz = []
        # long -> short
        for idx in range(tot):
            cur = dataset.token_reader[idx]
            if idx == 0:
                if cur["dur"] > max_batch_size:
                    raise ValueError("batch_size is smaller than maximum "
                                     "length of the utterances")
            utt_dur = cur["dur"]
            if cur_dur < max_batch_size:
                cur_dur += utt_dur
            else:
                idx_bz.append((beg, idx))
                cur_dur = utt_dur
                beg = idx
        if tot - beg > 1:
            idx_bz.append((beg, tot))
        return idx_bz

    def _work_adapt_batch_index(self, dataset: dat.Dataset, adapt_dur: float,
                                adapt_num: int, max_batch_size: int,
                                min_batch_size: int) -> List[Tuple[int, int]]:
        """
        In adaptive mode, we compute #batch_size using
            cur_bz = int(max(#min_batch_size, #max_batch_size // (1 + factor)))
        where:
            factor = max(cur_ilen // #adapt_dur, (cur_olen - 1) // #adapt_num)
        """
        beg = 0
        tot = len(dataset)
        cur_bz = max_batch_size
        idx_boundary = []
        while beg < tot:
            cur = dataset.token_reader[beg]
            cur_ilen = cur["dur"]
            cur_olen = cur["len"]
            factor = max(cur_ilen // adapt_dur, (cur_olen - 1) // adapt_num)
            cur_bz = int(max(min_batch_size, max_batch_size // (1 + factor)))
            idx_boundary.append((beg, min(beg + cur_bz, tot)))
            beg += cur_bz
        return idx_boundary

    def __iter__(self):
        indices = derive_indices(self.num_batches,
                                 seed=self.epoch,
                                 shuffle=self.shuffle,
                                 distributed=self.distributed)
        subset = [self.batches[i] for i in indices]
        return iter([list(range(beg, end)) for beg, end in subset])

    def set_epoch(self, epoch: int) -> NoReturn:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches


class ASRDataLoader(dat.DataLoader):
    """
    ASR dataloader for E2E AM training

    Args:
        dataset: instance of dat.Dataset
        collate_fn: collate function used in dat.DataLoader
        shuffle: shuffle batches or not
        distributed: in distributed mode or not
        num_workers: number of the workers used in dat.DataLoader
        adapt_dur|adapt_token_num: used in adaptive mode dataloader
        batch_mode: adaptive or constraint
        max_batch_size: maximum #batch_size
        min_batch_size: minimum #batch_size
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 collate_fn: Callable,
                 shuffle: bool = True,
                 distributed: bool = False,
                 num_workers: int = 0,
                 adapt_dur: float = 800,
                 adapt_token_num: int = 150,
                 batch_mode: str = "adaptive",
                 max_batch_size: int = 32,
                 min_batch_size: int = 4) -> None:
        sampler = BatchSampler(dataset,
                               max_batch_size,
                               shuffle=shuffle,
                               adapt_dur=adapt_dur,
                               batch_mode=batch_mode,
                               distributed=distributed,
                               min_batch_size=min_batch_size,
                               adapt_token_num=adapt_token_num)
        super(ASRDataLoader, self).__init__(dataset,
                                            collate_fn=collate_fn,
                                            num_workers=num_workers,
                                            batch_sampler=sampler)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
