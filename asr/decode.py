#!/usr/bin/env python

import sys
import yaml
import codecs
import pathlib
import argparse

import torch as th

from libs.evaluator import Evaluator
from libs.utils import get_logger, io_wrapper, StrToBoolAction
from loader.wav_loader import WaveReader
from nn import support_nnet
from feats import support_transform
from kaldi_python_io import ScriptReader

logger = get_logger(__name__)
"""
Nbest format:
Number: n
key1
score-1 hyp-1
...
score-n hyp-n
...
keyM
score-1 hyp-1
...
score-n hyp-n
"""


class FasterDecoder(Evaluator):
    """
    Decoder wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        nnet = self._load(cpt_dir)
        super(FasterDecoder, self).__init__(nnet, cpt_dir, device_id=device_id)

    def compute(self, src, **kwargs):
        src = th.from_numpy(src).to(self.device)
        return self.nnet.beam_search(src, **kwargs)

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
    # build dictionary
    if args.dict:
        with codecs.open(args.dict, "r", encoding="utf-8") as f:
            vocab = {}
            for pair in f:
                unit, idx = pair.split()
                vocab[int(idx)] = unit
    else:
        vocab = None
    decoder = FasterDecoder(args.checkpoint, device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    stdout_top1, top1 = io_wrapper(args.best, "w")
    topn = None
    if args.dump_nbest:
        stdout_topn, topn = io_wrapper(args.dump_nbest, "w")
        topn.write(f"{args.nbest}\n")

    N = 0
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest_hypos = decoder.compute(src,
                                      beam=args.beam_size,
                                      nbest=args.nbest,
                                      max_len=args.max_len,
                                      normalized=args.normalized,
                                      vectorized=args.vectorized)
        nbest = [f"{key}\n"]
        for idx, hyp in enumerate(nbest_hypos):
            score = hyp["score"]
            # remove SOS/EOS
            if vocab:
                trans = [vocab[idx] for idx in hyp["trans"][1:-1]]
            else:
                trans = [str(idx) for idx in hyp["trans"][1:-1]]
            if vocab and args.space:
                trans = "".join(trans).replace(args.space, " ")
            else:
                trans = " ".join(trans)
            nbest.append(f"{score:.3f}\t{trans}\n")
            if idx == 0:
                top1.write(f"{key}\t{trans}\n")
        if topn:
            topn.write("".join(nbest))
        if not (N + 1) % 50:
            top1.flush()
            if topn:
                topn.flush()
        N += 1
    if not stdout_top1:
        top1.close()
    if topn and not stdout_topn:
        topn.close()
    logger.info(f"Decode {len(src_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do End-To-End decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("best",
                        type=str,
                        help="Wspecifier for decoded results (1-best)")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the E2E model")
    parser.add_argument("--beam-size",
                        type=int,
                        default=8,
                        help="Beam size used during decoding")
    parser.add_argument("--dict",
                        type=str,
                        default="",
                        help="Dictionary file (not needed)")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--max-len",
                        type=int,
                        default=100,
                        help="Maximum steps to do during decoding stage")
    parser.add_argument("--space",
                        type=str,
                        default="",
                        help="space flag for language like EN "
                        "to merge characters to words")
    parser.add_argument("--nbest",
                        type=int,
                        default=1,
                        help="N-best decoded utterances to output")
    parser.add_argument("--dump-nbest",
                        type=str,
                        default="",
                        help="If not empty, dump n-best hypothesis")
    parser.add_argument("--nnet",
                        type=str,
                        default="las",
                        choices=["las", "enh_las", "transformer"],
                        help="Network type used")
    parser.add_argument("--vectorized",
                        action=StrToBoolAction,
                        default="false",
                        help="If ture, using vectorized algothrim")
    parser.add_argument("--normalized",
                        action=StrToBoolAction,
                        default="false",
                        help="If ture, using length normalized "
                        "when sort nbest hypos")
    args = parser.parse_args()
    run(args)
