#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import codecs

from typing import List
from kaldi_python_io import Reader as BaseReader


class TextReader(BaseReader):
    """
    Reader for Kaldi's text file
    """

    def __init__(self, text: str, char: bool = False):
        super(TextReader, self).__init__(text, num_tokens=-1, restrict=False)
        self.char = char

    def _load(self, key) -> List[str]:
        """
        Return character or word sequence
        """
        words = self.index_dict[key]
        if self.char:
            chars = []
            for word in words:
                chars += [c for c in word]
            return chars
        else:
            return words


class NbestReader(object):
    """
    N-best hypothesis reader
    """

    def __init__(self, nbest: str):
        self.nbest, self.hypos = self._load_nbest(nbest)

    def _load_nbest(self, nbest: str):
        hypos = {}
        nbest = 1
        with codecs.open(nbest, "r", encoding="utf-8") as f:
            nbest = int(f.readline())
            while True:
                key = f.readline().strip()
                if not key:
                    break
                topk = []
                n = 0
                while n < self.nbest:
                    items = f.readline().strip().split()
                    score = float(items[0])
                    num_tokens = int(items[1])
                    trans = " ".join(items[2:])
                    topk.append((score, num_tokens, trans))
                    n += 1
                hypos[key] = topk
        return nbest, hypos

    def __iter__(self):
        for key in self.hypos:
            yield key, self.hypos[key]
