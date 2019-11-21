#!/usr/bin/env python

import yaml
import random
import pathlib
import argparse

import torch as th

from torch.nn.utils.rnn import pad_sequence

from libs.evaluator import Evaluator
from libs.utils import get_logger, StrToBoolAction
from loader.wav_loader import WaveReader
from nn import support_nnet
from feats import support_transform
from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class BatchDecoder(Evaluator):
    """
    Decoder wrapper
    """
    def __init__(self, cpt_dir, device_id=-1):
        nnet = self._load(cpt_dir)
        super(BatchDecoder, self).__init__(nnet, cpt_dir, device_id=-1)

    def compute(self, src, xlen, **kwargs):
        src = pad_sequence([th.from_numpy(s) for s in src],
                           batch_first=True,
                           padding_value=0)
        src = src.to(self.device)
        xlen = th.tensor(xlen, dtype=th.int64, device=self.device)
        return self.nnet.beam_search_batch(src, xlen, **kwargs)

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
    if args.batch_size == 1:
        raise RuntimeError("batch_size == 1, use decode.py instead")
    # build dictionary
    if args.dict:
        with open(args.dict, "r") as f:
            vocab = {}
            for pair in f:
                unit, idx = pair.split()
                vocab[int(idx)] = unit
    else:
        vocab = None
    decoder = BatchDecoder(args.checkpoint, device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    logger.info("Prepare dataset ...")
    # long to short
    utts_sort = [{"key": key, "len": src.shape[-1]} for key, src in src_reader]
    utts_sort = sorted(utts_sort, key=lambda n: n["len"], reverse=True)
    logger.info("Prepare dataset done")
    batches = []
    n = 0
    while n < len(utts_sort):
        batches.append(utts_sort[n:n + args.batch_size])
        n += args.batch_size
    random.shuffle(batches)

    output = open(args.output, "w") if args.output != "-" else None

    for batch in batches:
        # prepare inputs
        keys = [b["key"] for b in batch]
        xlen = [b["len"] for b in batch]
        mats = [src_reader[key] for key in keys]
        # decode
        batch_nbest = decoder.compute(mats,
                                      xlen,
                                      beam=args.beam_size,
                                      nbest=args.nbest,
                                      max_len=args.max_len,
                                      normalized=args.normalized)

        for key, nbest in zip(keys, batch_nbest):
            logger.info(f"Decoding utterance {key}...")
            for hyp in nbest:
                score = hyp["score"]
                logger.info(f"{key} score = {score:.2f}")
                # remove SOS/EOS
                if vocab:
                    trans = [vocab[idx] for idx in hyp["trans"][1:-1]]
                else:
                    trans = [str(idx) for idx in hyp["trans"][1:-1]]
                if vocab and args.space:
                    trans = "".join(trans).replace(args.space, " ")
                else:
                    trans = " ".join(trans)
                if not output:
                    print(f"{key} {trans}", flush=True)
                else:
                    output.write(f"{key} {trans}\n")
        output.flush()
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
    parser.add_argument("--nnet",
                        type=str,
                        default="las",
                        choices=["las", "enh_las", "transformer"],
                        help="Network type used")
    parser.add_argument("--batch-size",
                        type=int,
                        default=4,
                        help="Number of utterances to process in a batch")
    parser.add_argument("--normalized",
                        action=StrToBoolAction,
                        default="false",
                        help="If ture, using length normalized "
                        "when sort nbest hypos")
    args = parser.parse_args()
    run(args)