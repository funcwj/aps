#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from abc import ABC

from aps.const import UNK_TOKEN
from aps.libs import Register
from typing import Dict, List, Union

ApsTokenizer = Register("tokenizer")


class TokenizerAbc(ABC):
    """
    ABC (abstract class) for tokenizer
    """

    def encode(self, utt: Union[str, List[str]]) -> List[str]:
        raise NotImplementedError

    def decode(self, utt: Union[str, List[str]]) -> List[str]:
        raise NotImplementedError


class Tokenizer(TokenizerAbc):
    """
    Mapping between string sequences & number/id sequences
    """

    def __init__(self,
                 vocab_dict: Dict,
                 tokenizer: str = "",
                 tokenizer_kwargs: Dict = {}):
        super(Tokenizer, self).__init__()
        if tokenizer:
            if tokenizer not in ApsTokenizer:
                raise ValueError(f"Unsupported tokenizer: {tokenizer}")
            self.tokenizer = ApsTokenizer[tokenizer](**tokenizer_kwargs)
        else:
            self.tokenizer = None
        if UNK_TOKEN in vocab_dict:
            self.unk_idx = vocab_dict[UNK_TOKEN]
        else:
            self.unk_idx = None
        # map str => int
        self.str2int = vocab_dict
        # map int => str
        self.int2str = {}
        for key, val in vocab_dict.items():
            self.int2str[val] = key

    def symbol2int(self, sym: str) -> int:
        """
        Return index id of the symbol
        """
        return self.str2int[sym]

    def int2symbol(self, idx: int) -> str:
        """
        Return string symbol of the index
        """
        return self.int2str[idx]

    def encode(self, str_seq: List[str]) -> List[int]:
        """
        Encode string sequences to int sequences
        """
        # if has tokenizer
        if self.tokenizer:
            str_seq = self.tokenizer.encode(str_seq)
        if self.unk_idx is None:
            return [self.str2int[c] for c in str_seq]
        else:
            return [(self.str2int[c] if c in self.str2int else self.unk_idx)
                    for c in str_seq]

    def decode(self, int_seq: List[int], unk_sym: str = "<unk>") -> List[str]:
        """
        Dncode int sequences to string sequences
        """
        str_seq = [self.int2str[n] for n in int_seq]
        if self.tokenizer:
            str_seq = self.tokenizer.decode(str_seq)
        # we have unk symbol
        if self.unk_idx is not None and unk_sym != UNK_TOKEN:
            return [(s if s != UNK_TOKEN else unk_sym) for s in str_seq]
        else:
            return str_seq
