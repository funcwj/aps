#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from typing import List, Union
from aps.tokenizer.base import TokenizerAbc, ApsTokenizer
from aps.tokenizer.word import WordTokenizer


@ApsTokenizer.register("subword")
class SubwordTokenizer(TokenizerAbc):
    """
    Class of the subword tokenizer (word pieces)
    Args:
        spm (str): path of the word piece model
        filter_words (list): words to be filtered
    """

    def __init__(self, spm: str = "", filter_words: List[str] = []):
        super(SubwordTokenizer, self).__init__()
        import sentencepiece as sp
        self.sp_mdl = sp.SentencePieceProcessor(model_file=spm)
        self.word_tokenizer = WordTokenizer(filter_words)

    def encode(self, word_seq: Union[str, List[str]]) -> List[str]:
        """
        Encode word string sequences to subword string sequences
        """
        words = self.word_tokenizer.encode(word_seq)
        return self.sp_mdl.encode(" ".join(words), out_type=str)

    def decode(self, subword_seq: Union[str, List[str]]) -> List[str]:
        """
        Encode subword string sequences to word string sequences
        """
        subwords = self.word_tokenizer.decode(subword_seq)
        return self.sp_mdl.decode(subwords).split(" ")
