# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import torch as th
import numpy as np

from typing import List
from itertools import permutations


class ChunkStitcher(object):
    """
    Stitcher for chunk style evaluation of the speech enhancement/separation models
    Reference:
        Chen Z, Yoshioka T, Lu L, et al. Continuous speech separation:
        dataset and analysis[C]//ICASSP 2020-2020 IEEE International Conference
        on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020: 7284-7288.
    """

    def __init__(self, chunk_len: int, lctx: int, rctx: int) -> None:
        self.chunk_len = chunk_len
        self.lctx, self.rctx = lctx, rctx

    def _reorder(self, pred: List[th.Tensor], succ: List[th.Tensor]):
        # no overlapped segment, skip
        if self.lctx == 0:
            return succ
        num_streams = len(pred)
        # overlapped segment, to work out distance
        pred_ov = [c[-self.lctx - self.rctx:] for c in pred]
        succ_ov = [c[:self.lctx + self.rctx] for c in succ]
        permu_list = list(permutations(range(num_streams)))
        dists = []
        for permu in permu_list:
            dists.append(
                sum([
                    th.abs(pred_ov[i] - succ_ov[j]).sum().item()
                    for i, j in enumerate(permu)
                ]))
        permu = permu_list[np.argmin(dists)]
        return [succ[i] for i in permu]

    def _stitch_one_stream(self, chunks: List[th.Tensor],
                           expected_length: int) -> th.Tensor:
        stream = th.zeros(expected_length)
        for i, chunk in enumerate(chunks):
            beg = i * self.chunk_len + self.lctx
            if i == 0:
                stream[:beg + self.chunk_len] = chunk[:beg + self.chunk_len]
            elif i == len(chunks) - 1:
                last_len = min(expected_length - beg,
                               chunk.shape[-1] - self.lctx)
                stream[beg:beg + last_len] = chunk[self.lctx:self.lctx +
                                                   last_len]
            else:
                stream[beg:beg + self.chunk_len] = chunk[self.lctx:self.lctx +
                                                         self.chunk_len]
        return stream

    def _stitch_multiple_streams(self, chunks: List[List[th.Tensor]],
                                 expected_length: int):
        num_streams = len(chunks[-1])
        stream_chunks = []
        # fix possible permutation problems
        for i, chunk in enumerate(chunks):
            if i:
                chunk = self._reorder(stream_chunks[-1], chunk)
            stream_chunks.append(chunk)
        return [
            self._stitch_one_stream([s[i]
                                     for s in stream_chunks], expected_length)
            for i in range(num_streams)
        ]

    def stitch(self, chunks: List, expected_length: int):
        num_streams = 1
        if not isinstance(chunks[-1], th.Tensor):
            num_streams = len(chunks[-1])
        if num_streams == 1:
            return self._stitch_one_stream(chunks, expected_length)
        else:
            return self._stitch_multiple_streams(chunks, expected_length)
