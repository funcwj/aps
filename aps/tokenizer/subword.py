#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from typing import List, Union
from aps.tokenizer.base import Tokenizer, WordTokenizer


class SubwordTokenizer(Tokenizer):
    """
    Class of the subword tokenizer (word pieces)
    Args:
        spm (str): path of the word piece model
        filter_words (list): words to be filtered
    """

    def __init__(self, spm: str, filter_words: List[str] = []):
        import sentencepiece as sp
        self.sp_mdl = sp.SentencePieceProcessor(model_file=spm)
        self.word_tokenizer = WordTokenizer(filter_words, char=False)

    def run(self, utt: Union[str, List[str]]) -> List[str]:
        words = self.word_tokenizer.run(utt)
        return self.sp_mdl.encode(" ".join(words), out_type=str)
