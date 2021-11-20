# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from aps.tokenizer import Tokenizer
from aps.conf import load_dict

from typing import List


class TextProcess(object):
    """
    Base class for pre-/post-processing of the audio transcriptions, i.e.,
    mapping them to the sequence ids for CTC alignment or mapping decoding
    id sequences to word sequences
    """

    def __init__(self, dict_str: str, space: str = "", spm: str = "") -> None:
        tokenizer_kwargs = {}
        if spm:
            tokenizer = "subword"
            tokenizer_kwargs["spm"] = spm
        else:
            if space:
                tokenizer = "char"
                tokenizer_kwargs["space"] = space
            else:
                tokenizer = "word"
        # str to int
        if dict_str:
            vocab_dict = load_dict(dict_str)
            self.tokenizer = Tokenizer(vocab_dict,
                                       tokenizer=tokenizer,
                                       tokenizer_kwargs=tokenizer_kwargs)
        else:
            self.tokenizer = None


class TextPreProcessor(TextProcess):
    """
    Text pre-processing class
    """

    def __init__(self, dict_str: str, space: str = "", spm: str = "") -> None:
        super(TextPreProcessor, self).__init__(dict_str, space=space, spm=spm)

    def run(self, str_seq: List[str]) -> List[int]:
        if self.tokenizer:
            int_seq = self.tokenizer.encode(str_seq)
        else:
            # no tokenizer avaiable
            int_seq = [int(idx) for idx in str_seq]
        return int_seq


class TextPostProcessor(TextProcess):
    """
    Text post-processing class
    """

    def __init__(self,
                 dict_str: str,
                 space: str = "",
                 show_unk: str = "<unk>",
                 spm: str = "") -> None:
        super(TextPostProcessor, self).__init__(dict_str, space=space, spm=spm)
        self.unk = show_unk

    def run(self, int_seq: List[int]) -> str:
        if self.tokenizer:
            str_seq = self.tokenizer.decode(int_seq, unk_sym=self.unk)
        else:
            # if tokenizer avaiable
            str_seq = [str(idx) for idx in int_seq]
        return " ".join(str_seq)
