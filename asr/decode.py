#!/usr/bin/env python

import argparse

import torch as th

from libs.evaluator import Evaluator
from libs.utils import get_logger
from loader.wav_loader import WaveReader
from feats.asr import FeatureTransform
from nn import support_nnet
from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class FasterDecoder(Evaluator):
    """
    Decoder wrapper
    """
    def __init__(self, *args, **kwargs):
        super(FasterDecoder, self).__init__(*args, **kwargs)

    def compute(self, src, **kwargs):
        src = th.from_numpy(src).to(self.device)
        return self.nnet.beam_search(src, **kwargs)


def run(args):
    # build dictionary
    if args.dict:
        with open(args.dict, "r") as f:
            vocab = {w: idx for w, idx in line.split() for line in f}
    else:
        vocab = None
    decoder = FasterDecoder(support_nnet(args.nnet),
                            FeatureTransform,
                            args.checkpoint,
                            device_id=args.device_id)
    if decoder.raw_waveform:
        src_reader = WaveReader(args.feats_or_wav_scp, sr=16000)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    output = open(args.output, "w") if args.output != "-" else None
    N = 0
    for key, src in src_reader:
        logger.info(f"Decoding utterance {key}...")
        nbest = decoder.compute(src,
                                beam=args.beam_size,
                                nbest=1,
                                max_len=args.max_len,
                                parallel=True)
        for token in nbest:
            score = token["score"]
            logger.info(f"{key}\tscore = {score:.2f}")
            if vocab:
                trans = [vocab[idx]
                         for idx in token["trans"][1:-1]]  # remove SOS/EOS
            else:
                trans = [str(idx) for idx in token["trans"][1:-1]]
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
    parser.add_argument("--nnet",
                        type=str,
                        default="common",
                        choices=["common", "transformer"],
                        help="Network type used")
    args = parser.parse_args()
    run(args)