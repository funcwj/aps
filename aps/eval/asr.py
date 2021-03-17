# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

from aps.conf import load_dict
from aps.const import UNK_TOKEN

from typing import List


class TextPreProcessor(object):
    """
    The class for pre processing of the transcriptions, i.e.,
    mapping them to the sequence ids for CTC alignment
    """

    def __init__(self, dict_str: str, space: str = "", spm: str = "") -> None:
        self.vocab = None
        self.space = space
        self.sp_mdl = None
        if dict_str:
            # str to int
            self.vocab = load_dict(dict_str)
        if spm:
            import sentencepiece as sp
            self.sp_mdl = sp.SentencePieceProcessor(model_file=spm)

    def run(self, str_seq: List[str]) -> List[int]:
        if self.vocab:
            if self.sp_mdl:
                # subword str sequence
                str_seq = self.sp_mdl.encode(" ".join(str_seq), out_type=str)
            if self.space:
                # insert space
                str_seq = f" {self.space} ".join(str_seq).split(" ")
            int_seq = [(self.vocab[idx]
                        if idx in self.vocab else self.vocab[UNK_TOKEN])
                       for idx in str_seq]
        else:
            int_seq = [int(idx) for idx in str_seq]
        return int_seq


class TextPostProcessor(object):
    """
    The class for post processing of decoding sequence, i.e.,
    mapping id sequence to token sequence
    """

    def __init__(self,
                 dict_str: str,
                 space: str = "",
                 show_unk: str = "<unk>",
                 spm: str = "") -> None:
        self.unk = show_unk
        self.space = space
        self.vocab = None
        self.sp_mdl = None
        if dict_str:
            # int to str
            self.vocab = load_dict(dict_str, reverse=True)
        if spm:
            import sentencepiece as sp
            self.sp_mdl = sp.SentencePieceProcessor(model_file=spm)

    def run(self, int_seq: List[int]) -> str:
        if self.vocab:
            trans = [self.vocab[idx] for idx in int_seq]
        else:
            trans = [str(idx) for idx in int_seq]
        # char sequence
        if self.vocab:
            if self.sp_mdl:
                trans = self.sp_mdl.decode(trans)
            else:
                if self.space:
                    trans = "".join(trans).replace(self.space, " ")
                else:
                    trans = " ".join(trans)
            if self.unk != UNK_TOKEN:
                trans = trans.replace(UNK_TOKEN, self.unk)
        # ID sequence
        else:
            trans = " ".join(trans)
        return trans
