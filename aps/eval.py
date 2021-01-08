# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pathlib

import torch as th
import torch.nn as nn

from aps.libs import aps_transform, aps_asr_nnet, aps_sse_nnet
from aps.conf import load_dict
from aps.const import UNK_TOKEN
from typing import Dict, List, Tuple


class NnetEvaluator(object):
    """
    A simple wrapper for model evaluation
    """

    def __init__(self,
                 cpt_dir: str,
                 cpt_tag: str = "best",
                 device_id: int = -1,
                 task: str = "asr") -> None:
        # load nnet
        self.epoch, self.nnet, self.conf = self._load(cpt_dir,
                                                      cpt_tag=cpt_tag,
                                                      task=task)
        # offload to device
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()

    def _load(self,
              cpt_dir: str,
              cpt_tag: str = "best",
              task: str = "asr") -> Tuple[int, nn.Module, Dict]:
        if task not in ["asr", "sse"]:
            raise ValueError(f"Unknown task name: {task}")
        cpt_dir = pathlib.Path(cpt_dir)
        # load checkpoint
        cpt = th.load(cpt_dir / f"{cpt_tag}.pt.tar", map_location="cpu")
        with open(cpt_dir / "train.yaml", "r") as f:
            conf = yaml.full_load(f)
            if task == "asr":
                net_cls = aps_asr_nnet(conf["nnet"])
            else:
                net_cls = aps_sse_nnet(conf["nnet"])
        asr_transform = None
        enh_transform = None
        self.accept_raw = False
        if "asr_transform" in conf:
            asr_transform = aps_transform("asr")(**conf["asr_transform"])
            self.accept_raw = True
        if "enh_transform" in conf:
            enh_transform = aps_transform("enh")(**conf["enh_transform"])
            self.accept_raw = True
        if enh_transform and asr_transform:
            nnet = net_cls(enh_transform=enh_transform,
                           asr_transform=asr_transform,
                           **conf["nnet_conf"])
        elif asr_transform:
            nnet = net_cls(asr_transform=asr_transform, **conf["nnet_conf"])
        elif enh_transform:
            nnet = net_cls(enh_transform=enh_transform, **conf["nnet_conf"])
        else:
            nnet = net_cls(**conf["nnet_conf"])

        nnet.load_state_dict(cpt["model_state"])
        return cpt["epoch"], nnet, conf

    def run(self, *args, **kwargs):
        raise NotImplementedError


class TextPostProcessor(object):
    """
    The class for post processing of decoding sequence
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
