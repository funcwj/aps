# wujian@2019

import yaml

import torch as th

from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)


class Evaluator(object):
    """
    A simple wrapper for model evaluation
    """
    def __init__(self, nnet, cpt_dir, device_id=-1):
        cpt_dir = Path(cpt_dir)
        # load checkpoint
        cpt = th.load(cpt_dir / "best.pt.tar", map_location="cpu")
        epoch = cpt["epoch"]
        # log state
        logger.info(f"Load model from checkpoint at {cpt_dir}/best.pt.tar " +
                    f"on epoch {epoch}")
        # load nnet
        self.nnet = nnet
        self.nnet.load_state_dict(cpt["model_state_dict"])
        if device_id < 0:
            self.device = th.device("cpu")
        else:
            self.device = th.device(f"cuda:{device_id:d}")
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()

    def compute(self, egs):
        raise NotImplementedError