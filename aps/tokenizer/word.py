#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from typing import List, Union
from aps.tokenizer.base import TokenizerAbc, ApsTokenizer


class WordBasedTokenizer(TokenizerAbc):
    """
    Word based (word, character) tokenizer
    Args:
        filter_words (list): filter those words
        char (bool): use character unit or word unit
        space (str): insert space symbol between words
    """

    def __init__(self,
                 filter_words: List[str] = [],
                 char: bool = False,
                 space: str = ""):
        super(WordBasedTokenizer, self).__init__()
        self.char = char
        self.space = space
        self.filter_words = filter_words

    def encode(self, utt: Union[str, List[str]]) -> List[str]:
        if isinstance(utt, str):
            raw_tokens = utt.split()
        else:
            raw_tokens = utt
        kept_tokens = []
        for tok in raw_tokens:
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
            if self.space:
                kept_tokens += [self.space]
        if self.space:
            # remove last one
            kept_tokens = kept_tokens[:-1]
        return kept_tokens

    def decode(self, utt: Union[str, List[str]]) -> List[str]:
        if isinstance(utt, str):
            enc_tokens = utt.split()
        else:
            enc_tokens = utt
        if not self.char:
            return enc_tokens
        if self.space:
            strs = "".join(enc_tokens).replace(self.space, " ")
        else:
            strs = " ".join(enc_tokens)
        return strs.split(" ")


@ApsTokenizer.register("word")
class WordTokenizer(WordBasedTokenizer):
    """
    Word tokenizer
    Args:
        filter_words (list): filter those words
    """

    def __init__(self, filter_words: List[str] = []):
        super(WordTokenizer, self).__init__(filter_words=filter_words,
                                            char=False,
                                            space="")


@ApsTokenizer.register("char")
class CharTokenizer(WordBasedTokenizer):
    """
    Character tokenizer
    Args:
        filter_words (list): filter those words
        space (str): insert space symbol between words
    """

    def __init__(self, filter_words: List[str] = [], space: str = "<space>"):
        super(CharTokenizer, self).__init__(filter_words=filter_words,
                                            char=True,
                                            space=space)
