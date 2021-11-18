#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC
from typing import List, Union
from aps.libs import Register

ApsTokenizer = Register("tokenizer")


class TokenizerAbc(ABC):
    """
    ABC (abstract class) for tokenizer
    """

    def run(self, utt: Union[str, List[str]]) -> List[str]:
        raise NotImplementedError


@ApsTokenizer.register("word")
class WordTokenizer(TokenizerAbc):
    """
    Word or character tokenizer
    Args:
        filter_words (list): filter those words
        char (bool): use character or word
        space (bool): insert space symbol between word or not
    """

    def __init__(self,
                 filter_words: List[str],
                 char: bool = False,
                 space: str = ""):
        super(WordTokenizer, self).__init__()
        self.char = char
        self.space = space
        self.filter_words = filter_words

    def run(self, utt: Union[str, List[str]]) -> List[str]:
        if isinstance(utt, str):
            raw_tokens = utt.split()
        else:
            raw_tokens = utt
        kept_tokens = []
        for n, tok in enumerate(raw_tokens):
            # remove tokens
            is_filter_tok = tok in self.filter_words
            if is_filter_tok:
                continue
            # word => char
            if self.char and not is_filter_tok:
                toks = [t for t in tok]
            else:
                toks = [tok]
            kept_tokens += toks
            if self.space and n != len(raw_tokens) - 1:
                kept_tokens += [self.space]
        return kept_tokens
