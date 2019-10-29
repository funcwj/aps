#!/usr/bin/env python

import argparse

import torch as th

from libs.evaluator import Evaluator
from libs.utils import get_logger

from kaldi_python_io import ScriptReader

from seq2seq import Seq2Seq

logger = get_logger(__name__)


class Decoder(Evaluator):
    """
    Decoder wrapper
    """
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def compute(self, feats, **kwargs):
        feats = th.from_numpy(feats).to(self.device)
        return self.nnet.beam_search(feats, **kwargs)


def run(args):

    feats_reader = ScriptReader(args.feats_scp)
    # build dictionary
    with open(args.dict, "rb") as f:
        token2idx = [line.decode("utf-8").split()[0] for line in f]
    vocab = {idx: char for idx, char in enumerate(token2idx)}

    decoder = Decoder(Seq2Seq, args.checkpoint, gpu_id=args.gpu)

    output = open(args.output, "w") if args.output != "-" else None
    N = 0
    for key, feats in feats_reader:
        logger.info("Decoding utterance {}...".format(key))
        nbest = decoder.compute(feats,
                                beam=args.beam_size,
                                nbest=1,
                                max_len=args.max_len)
        for token in nbest:
            logger.info("{}\tscore = {:.2f}".format(key, token["score"]))
            trans = [vocab[idx]
                     for idx in token["trans"][1:-1]]  # remove SOS/EOS
            if not output:
                print("{}\t{}".format(key, " ".join(trans)), flush=True)
            else:
                output.write("{}\t{}\n".format(key, " ".join(trans)))
                if not (N + 1) % 100:
                    output.flush()
        N += 1
    if output:
        output.close()
    logger.info("Decode {:d} utterance done".format(len(feats_reader)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do End-To-End decoding using beam search algothrim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint",
                        type=str,
                        help="Checkpoint of the E2E model")
    parser.add_argument("feats_scp",
                        type=str,
                        help="Rspecifier for evaluation feature files")
    parser.add_argument("output",
                        type=str,
                        help="Wspecifier for decoded results")
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
                        required=True,
                        help="Dictionary file")
    parser.add_argument("--gpu",
                        type=int,
                        default=-1,
                        help="GPU-id to offload model to, "
                        "-1 means running on CPU")
    args = parser.parse_args()
    run(args)