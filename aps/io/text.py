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

    def __len__(self) -> int:
        return len(self.hypos)

    def __iter__(self):
        return iter(self.hypos.items())

    def _load_nbest(self, nbest: str):
        hypos = {}
        with codecs.open(nbest, "r", encoding="utf-8") as fd:
            all_lines = fd.readlines()
        nbest = int(all_lines[0].strip())
        if (len(all_lines) - 1) % (nbest + 1) != 0:
            raise RuntimeError("Seems that nbest format is wrong")
        n = 1
        while n < len(all_lines):
            key = all_lines[n].strip()
            topk = []
            for i in range(nbest):
                items = all_lines[n + 1 + i].strip().split()
                score = float(items[0])
                num_tokens = int(items[1])
                trans = " ".join(items[2:])
                topk.append((score, num_tokens, trans))
            n += nbest + 1
            hypos[key] = topk
        return nbest, hypos
