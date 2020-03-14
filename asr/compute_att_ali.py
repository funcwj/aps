#!/usr/bin/env python

import yaml
import pathlib
import argparse

import torch as th
import numpy as np

from libs.evaluator import Evaluator
from libs.utils import get_logger
from loader.wave import WaveReader

from kaldi_python_io import ScriptReader
from kaldi_python_io import Reader as BaseReader

from nn import support_nnet
from feats import support_transform

logger = get_logger(__name__)


class Computer(Evaluator):
    """
    Alignments computation wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        nnet = self._load(cpt_dir)
        super(Computer, self).__init__(nnet, cpt_dir, device_id=device_id)

    def compute(self, src, token):
        src = th.from_numpy(src).to(self.device)
        token = th.tensor(token, dtype=th.int64, device=self.device)
        prob, alis, _, _ = self.nnet(src[None, :], None, token[None, :], ssr=1)
        pred = th.argmax(prob.detach().squeeze(0), -1)[:-1]
        accu = th.sum(pred == token).float() / token.size(-1)
        logger.info(f"Accu = {accu.item():.2f}")
        return alis.detach().cpu().squeeze().numpy()

    def _load(self, cpt_dir):
        with open(pathlib.Path(cpt_dir) / "train.yaml", "r") as f:
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
        return nnet


def run(args):
    token_reader = BaseReader(args.token,
                              value_processor=lambda l: [int(n) for n in l],
                              num_tokens=-1,
                              restrict=False)
    computer = Computer(args.checkpoint, device_id=args.device_id)
    if computer.accept_raw:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    dump_dir = pathlib.Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    for key, src in src_reader:
        logger.info(f"Processing utterance {key}...")
        alis = computer.compute(src, token_reader[key])
        np.save(dump_dir / key, alis)
    logger.info(f"Processed {len(src_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to compute attention alignments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("token",
                        type=str,
                        help="Rspecifier for evaluation transcriptions")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the acoustic model")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--dump-dir",
                        type=str,
                        default="att_ali",
                        help="Output directory for alignments")
    args = parser.parse_args()
    run(args)