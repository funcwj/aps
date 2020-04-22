# wujian@2019

import yaml
import pathlib

import torch as th

from .nn import support_nnet
from .feats import support_transform


class Computer(object):
    """
    A simple wrapper for model evaluation
    """
    def __init__(self, nnet, cpt_dir, device_id=-1):
        # load nnet
        self.epoch, self.nnet, self.conf = self._load(cpt_dir)
        # offload to device
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()

    def _load(self, cpt_dir):
        cpt_dir = pathlib.Path(cpt_dir)
        # load checkpoint
        cpt = th.load(cpt_dir / "best.pt.tar", map_location="cpu")
        with open(cpt_dir / "train.yaml", "r") as f:
            conf = yaml.full_load(f)
            asr_cls = support_nnet(conf["nnet_type"])
        asr_transform = None
        enh_transform = None
        self.accept_raw = False
        if "asr_transform" in conf:
            asr_transform = support_transform("asr")(**conf["asr_transform"])
            self.accept_raw = True
        if "enh_transform" in conf:
            enh_transform = support_transform("enh")(**conf["enh_transform"])
            self.accept_raw = True
        if enh_transform:
            nnet = asr_cls(enh_transform=enh_transform,
                           asr_transform=asr_transform,
                           **conf["nnet_conf"])
        elif asr_transform:
            nnet = asr_cls(asr_transform=asr_transform, **conf["nnet_conf"])
        else:
            nnet = asr_cls(**conf["nnet_conf"])

        nnet.load_state_dict(cpt["model_state_dict"])
        return cpt["epoch"], nnet, conf

    def run(self, *args, **kwargs):
        raise NotImplementedError