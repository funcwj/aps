#!/usr/bin/env python

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from aps.tokenizer.base import Tokenizer
from aps.const import UNK_TOKEN

from typing import Dict, List, Optional


class Vocab(object):
    """
    Mapping token sequence to number/id sequence
    """

    def __init__(self, vocab_dict: Dict, tokenizer: Optional[Tokenizer] = None):
        self.vocab_dict = vocab_dict
        self.tokenizer = tokenizer

    def __call__(self, tokens: List[str]) -> List[int]:
        # if has tokenizer
        if self.tokenizer:
            tokens = self.tokenizer.run(tokens)
        return [(self.vocab_dict[t]
                 if t in self.vocab_dict else self.vocab_dict[UNK_TOKEN])
                for t in tokens]
