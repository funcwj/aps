# wujian@2019

import yaml
import os.path as op

import torch as th

from .logger import get_logger

logger = get_logger(__name__)


class Evaluator(object):
    """
    A simple wrapper for model evaluation
    """

    def __init__(self, nnet_cls, cpt_dir, gpu_id=-1):
        # load nnet
        self.nnet = self._load_nnet(nnet_cls, cpt_dir)
        self.device = th.device(
            "cpu" if gpu_id < 0 else "cuda:{:d}".format(gpu_id))
        if gpu_id >= 0:
            self.nnet.to(self.device)
        # set eval model
        self.nnet.eval()

    def compute(self, egs):
        raise NotImplementedError

    def _load_nnet(self, nnet_cls, cpt_dir):
        """
        Load model from checkpoints
        """
        with open(op.join(cpt_dir, "train.yaml"), "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            nnet = nnet_cls(**conf["nnet_conf"])
        # load checkpoint
        cpt_fname = op.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        # log state
        logger.info("Load model from checkpoint at {}, on epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet