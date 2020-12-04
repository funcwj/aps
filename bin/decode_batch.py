#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import warnings

import torch as th

from torch.nn.utils.rnn import pad_sequence

from aps.loader import AudioReader
from aps.utils import get_logger, io_wrapper
from aps.opts import StrToBoolAction
from aps.conf import load_dict
from aps.eval import Computer

from kaldi_python_io import ScriptReader

logger = get_logger(__name__)


class BatchDecoder(Computer):
    """
    Decoder wrapper
    """

    def __init__(self, cpt_dir, device_id=-1):
        super(BatchDecoder, self).__init__(cpt_dir, device_id=device_id)
        logger.info(f"Load checkpoint from {cpt_dir}: epoch {self.epoch}")

    def run(self, inp, inp_len, **kwargs):
        inp = pad_sequence([th.from_numpy(s) for s in inp],
                           batch_first=True,
                           padding_value=0).to(self.device)
        inp_len = th.tensor(inp_len, device=self.device)
        return self.nnet.beam_search_batch(inp, inp_len, **kwargs)


def run(args):
    if args.batch_size == 1:
        warnings.warn("can use decode.py instead as batch_size == 1")
    # build dictionary
    if args.dict:
        vocab = load_dict(args.dict, reverse=True)
    else:
        vocab = None
    decoder = BatchDecoder(args.checkpoint, device_id=args.device_id)
    if decoder.accept_raw:
        src_reader = AudioReader(args.feats_or_wav_scp,
                                 sr=args.sr,
                                 channel=args.channel)
    else:
        src_reader = ScriptReader(args.feats_or_wav_scp)

    logger.info("Prepare dataset ...")
    # long to short
    utts_sort = [{"key": key, "len": src.shape[-1]} for key, src in src_reader]
    utts_sort = sorted(utts_sort, key=lambda n: n["len"], reverse=True)
    logger.info(f"Prepare dataset done ({len(utts_sort)}) utterances")
    batches = []
    n = 0
    while n < len(utts_sort):
        batches.append(utts_sort[n:n + args.batch_size])
        n += args.batch_size
    if n != len(utts_sort):
        batches.append(utts_sort[n:])

    if args.lm:
        lm = Computer(args.lm, device_id=args.device_id)
        logger.info(f"Load lm from {args.lm}: epoch {lm.epoch}, " +
                    f"weight = {args.lm_weight}")
        lm = lm.nnet
    else:
        lm = None

    stdout_top1, top1 = io_wrapper(args.best, "w")
    topn = None
    if args.dump_nbest:
        stdout_topn, topn = io_wrapper(args.dump_nbest, "w")
        topn.write(f"{args.nbest}\n")

    for batch in batches:
        # prepare inputs
        keys = [b["key"] for b in batch]
        inp = [src_reader[key] for key in keys]
        inp_len = [b["len"] for b in batch]
        # decode
        batch_nbest = decoder.run(inp,
                                  inp_len,
                                  lm=lm,
                                  beam=args.beam_size,
                                  nbest=args.nbest,
                                  max_len=args.max_len,
                                  lm_weight=args.lm_weight,
                                  normalized=args.normalized)
        for key, nbest in zip(keys, batch_nbest):
            logger.info(f"Decoding utterance {key}...")
            nbest_hypos = [f"{key}\n"]
            for idx, hyp in enumerate(nbest):
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
                nbest_hypos.append(f"{score:.3f}\t{trans}\n")
                if idx == 0:
                    top1.write(f"{key}\t{trans}\n")
            if topn:
                topn.write("".join(nbest_hypos))
        top1.flush()
        if topn:
            topn.flush()
    if not stdout_top1:
        top1.close()
    if topn and not stdout_topn:
        topn.close()
    logger.info(f"Decode {len(src_reader)} utterance done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do End-To-End decoding using batch beam search algothrim "
        "(numbers may different when #batch_size != 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("feats_or_wav_scp",
                        type=str,
                        help="Feature/Wave scripts")
    parser.add_argument("best", type=str, help="Wspecifier for decoded results")
    parser.add_argument("--channel",
                        type=int,
                        default=-1,
                        help="Channel index for wav.scp")
    parser.add_argument("--sr",
                        type=int,
                        default=16000,
                        help="Sample rate of the original audio")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Checkpoint of the E2E model")
    parser.add_argument("--beam-size",
                        type=int,
                        default=8,
                        help="Beam size used during decoding")
    parser.add_argument("--lm",
                        type=str,
                        default="",
                        help="Checkpoint of the nerual network LM "
                        "used in shallow fusion")
    parser.add_argument("--lm-weight",
                        type=float,
                        default=0.1,
                        help="LM score weight used in shallow fusion")
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
