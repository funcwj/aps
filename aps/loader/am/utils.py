#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torch as th

import torch.utils.data as dat
import aps.distributed as dist

from typing import Dict, List, Tuple, NoReturn, Optional, Callable
from kaldi_python_io import Reader as BaseReader
from aps.const import UNK_TOKEN


class AsrDataset(dat.Dataset):
    """
    A base dataset class for AM training

    Args:
        input_reader: source feature reader instance
        text: path of the token file
        utt2dur: path of the duration file
        vocab_dict: vocabulary dictionary object
        {min,max}_token_num: filter the utterances if the token number not in [#min_token_num, #max_token_num]
        {min|max}_dur: filter the utterances when length is not in [#min_wav_dur, #max_wav_dur]
        skip_utts: skips utterances that the file shows
        duration_axis: duration axis index for input_reader
    """

    def __init__(self,
                 input_reader,
                 text: str,
                 utt2dur: str,
                 vocab_dict: Optional[Dict],
                 max_token_num: int = 400,
                 min_token_num: int = 2,
                 max_dur: float = 3000,
                 min_dur: float = 40,
                 skip_utts: str = "",
                 duration_axis=-1):
        self.input_reader = input_reader
        self.token_reader = TokenReader(text,
                                        utt2dur,
                                        vocab_dict,
                                        skip_utts=skip_utts,
                                        max_dur=max_dur,
                                        min_dur=min_dur,
                                        max_token_num=max_token_num,
                                        min_token_num=min_token_num)
        self.duration_axis = duration_axis

    def __getitem__(self, idx: int) -> Dict:
        tok = self.token_reader[idx]
        key = tok["key"]
        inp = self.input_reader[key]
        return {
            "dur": inp.shape[self.duration_axis],
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
        self.vocab_dict = vocab_dict
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
            L = len(tokens)
            if L > max_token_num or L <= min_token_num:
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
                "len": L,
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
            stats["tok"] = [(self.vocab_dict[t] if t in self.vocab_dict else
                             self.vocab_dict[UNK_TOKEN]) for t in stats["tok"]]
            stats["vis"] = True
        return stats

    def __len__(self) -> int:
        return len(self.token_list)


class BatchSampler(dat.Sampler):
    """
    A custom batch sampler that can used in distributed/non-distributed mode
    Args:
        dataset: dataset object
        batch_size: maxnimum #batch_size
        shuffle: shuffle batches or not
        batch_mode: "adaptive" or "constraint"
        adapt_dur|adapt_token_num: used in adaptive mode, see _work_adapt_batch_index
        min_batch_size: minimum #batch_size
        distributed: distributed or not
    """

    def __init__(self,
                 dataset: dat.Dataset,
                 batch_size: int,
                 shuffle: bool = False,
                 batch_mode: str = "adaptive",
                 adapt_dur: float = 800,
                 adapt_token_num: int = 150,
                 min_batch_size: int = 4,
                 distributed: bool = False) -> None:
        if batch_mode not in ["adaptive", "constraint"]:
            raise ValueError(f"Unsupported batch mode: {batch_mode}")
        self.distributed = distributed
        if distributed:
            self.world_size = dist.world_size()
            self.rank = dist.rank()
        if batch_mode == "adaptive":
            batches = self._work_adapt_batch_index(
                dataset,
                adapt_dur,
                adapt_token_num,
                batch_size,
                min_batch_size=min_batch_size)
        else:
            batches = self._work_const_batch_index(dataset, batch_size)
        if distributed:
            self.num_batches = len(batches) // self.world_size
        else:
            self.num_batches = len(batches)
        self.batches = batches
        self.genfunc = th.randperm if shuffle else th.arange
        self.epoch = 0

    def _work_const_batch_index(self, dataset: dat.Dataset,
                                batch_size: int) -> List[Tuple[int, int]]:
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
                if cur["dur"] > batch_size:
                    raise ValueError("batch_size is smaller than maximum "
                                     "length of the utterances")
            utt_dur = cur["dur"]
            if cur_dur < batch_size:
                cur_dur += utt_dur
            else:
                idx_bz.append((beg, idx))
                cur_dur = utt_dur
                beg = idx
        if tot - beg > 1:
            idx_bz.append((beg, tot))
        return idx_bz

    def _work_adapt_batch_index(
            self,
            dataset: dat.Dataset,
            adapt_dur: float,
            adapt_num: int,
            batch_size: int,
            min_batch_size: int = 4) -> List[Tuple[int, int]]:
        """
        In adaptive mode, we compute #batch_size using
            cur_bz = int(max(#min_batch_size, #batch_size // (1 + factor)))
        where:
            factor = max(cur_ilen // #adapt_dur, (cur_olen - 1) // #adapt_num)
        """
        beg = 0
        tot = len(dataset)
        cur_bz = batch_size
        idx_bz = []
        while beg + cur_bz <= tot:
            cur = dataset.token_reader[beg]
            cur_ilen = cur["dur"]
            cur_olen = cur["len"]
            factor = max(cur_ilen // adapt_dur, (cur_olen - 1) // adapt_num)
            cur_bz = int(max(min_batch_size, batch_size // (1 + factor)))
            idx_bz.append((beg, beg + cur_bz))
            beg += cur_bz
        return idx_bz

    def __iter__(self):
        if self.distributed:
            # deterministically shuffle based on epoch
            g = th.Generator()
            g.manual_seed(self.epoch)
            N = self.num_batches * self.world_size
            indices = th.randperm(N, generator=g).tolist()
            indices = indices[self.rank:N:self.world_size]
        else:
            indices = self.genfunc(self.num_batches).tolist()
        for i in indices:
            yield list(range(*self.batches[i]))

    def set_epoch(self, epoch: int) -> NoReturn:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches


class AsrDataLoader(dat.DataLoader):
    """
    ASR dataloader for E2E AM training

    Args:
        dataset: instance of dat.Dataset
        collate_fn: collate function used in dat.DataLoader
        shuffle: shuffle batches or not
        distributed: in distributed mode or not
        num_workers: number of the workers used in dat.DataLoader
        adapt_dur|adapt_token_num: used in adaptive mode dataloader
        batch_size: maximum #batch_size
        batch_mode: adaptive or constraint
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
                 batch_size: int = 32,
                 batch_mode: str = "adaptive",
                 min_batch_size: int = 4) -> None:
        sampler = BatchSampler(dataset,
                               batch_size,
                               shuffle=shuffle,
                               batch_mode=batch_mode,
                               distributed=distributed,
                               adapt_dur=adapt_dur,
                               adapt_token_num=adapt_token_num,
                               min_batch_size=min_batch_size)
        super(AsrDataLoader, self).__init__(dataset,
                                            batch_sampler=sampler,
                                            collate_fn=collate_fn,
                                            num_workers=num_workers)

    def set_epoch(self, epoch: int) -> NoReturn:
        self.batch_sampler.set_epoch(epoch)
