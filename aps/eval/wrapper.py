# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pathlib

import torch as th

from aps.libs import aps_transform, aps_nnet
from aps.utils import get_logger
from typing import Dict

logger = get_logger(__name__)


def load_checkpoint(cpt_dir: str, cpt_tag: str = "best") -> Dict:
    """
    Load well trained checkpoint
    """
    cpt_dir = pathlib.Path(cpt_dir)
    # load checkpoint
    cpt = th.load(cpt_dir / f"{cpt_tag}.pt.tar", map_location="cpu")
    with open(cpt_dir / "train.yaml", "r") as f:
        conf = yaml.full_load(f)
    nnet_cls = aps_nnet(conf["nnet"])
    asr_transform = None
    enh_transform = None
    accept_raw = False
    if "asr_transform" in conf:
        asr_transform = aps_transform("asr")(**conf["asr_transform"])
        # if no STFT layer
        accept_raw = asr_transform.spectra_index != -1
    if "enh_transform" in conf:
        enh_transform = aps_transform("enh")(**conf["enh_transform"])
        accept_raw = True
    if enh_transform and asr_transform:
        nnet = nnet_cls(enh_transform=enh_transform,
                        asr_transform=asr_transform,
                        **conf["nnet_conf"])
    elif asr_transform:
        nnet = nnet_cls(asr_transform=asr_transform, **conf["nnet_conf"])
    elif enh_transform:
        nnet = nnet_cls(enh_transform=enh_transform, **conf["nnet_conf"])
    else:
        nnet = nnet_cls(**conf["nnet_conf"])

    nnet.load_state_dict(cpt["model_state"])
    return {
        "epoch": cpt["epoch"],
        "accept_raw": accept_raw,
        "nnet": nnet,
        "conf": conf
    }


class NnetEvaluator(object):
    """
    A simple wrapper for the model evaluation
    """

    def __init__(self,
                 cpt_dir: str,
                 cpt_tag: str = "best",
                 device_id: int = -1) -> None:
        # load nnet
        stats = load_checkpoint(cpt_dir, cpt_tag=cpt_tag)
        self.conf = stats["conf"]
        self.nnet = stats["nnet"]
        self.accept_raw = stats["accept_raw"]
        # offload to device
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()
        # logging
        logger.info(f"Load the checkpoint from {cpt_dir}, epoch: " +
                    f"{stats['epoch']}, tag: {cpt_tag}, device: {device_id}")

    def run(self, *args, **kwargs):
        raise NotImplementedError
