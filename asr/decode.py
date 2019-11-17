#!/usr/bin/env python

import yaml
import pathlib
import argparse

import torch as th

from libs.evaluator import Evaluator
from libs.utils import get_logger, StrToBoolAction
from loader.wav_loader import WaveReader
from nn import support_nnet
from feats import support_transform
from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class FasterDecoder(Evaluator):
    """
    Decoder wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        nnet = self._load(cpt_dir)
        print(f"Nnet structure:\n{nnet}")
        super(FasterDecoder, self).__init__(nnet, cpt_dir, device_id=-1)

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
        with open(args.dict, "r") as f:
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

    output = open(args.output, "w") if args.output != "-" else None
    N = 0
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest = decoder.compute(src,
                                beam=args.beam_size,
                                nbest=args.nbest,
                                max_len=args.max_len,
                                normalized=args.normalized,
                                vectorized=args.vectorized)
        for hyp in nbest:
            score = hyp["score"]
            logger.info(f"{key}\tscore = {score:.2f}")
            # remove SOS/EOS
            if vocab:
                trans = [vocab[idx] for idx in hyp["trans"][1:-1]]
            else:
                trans = [str(idx) for idx in hyp["trans"][1:-1]]
            trans = " ".join(trans)
            if not output:
                print(f"{key}\t{trans}", flush=True)
            else:
                output.write(f"{key}\t{trans}\n")
                if not (N + 1) % 50:
                    output.flush()
        N += 1
    if output:
        output.close()
    logger.info(f"Decode {len(src_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do End-To-End decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("output",
                        type=str,
                        help="Wspecifier for decoded results")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the E2E model")
    parser.add_argument("--max-len",
                        type=int,
                        default=100,
                        help="Maximum steps to do during decoding stage")
    parser.add_argument("--beam-size",
                        type=int,
                        default=8,
                        help="Beam size used during decoding")
    parser.add_argument("--dict",
                        type=str,
                        help="Dictionary file (not needed)")
    parser.add_argument("--device-id",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    parser.add_argument("--nbest",
                        type=int,
                        default=1,
                        help="N-best decoded utterances to output")
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