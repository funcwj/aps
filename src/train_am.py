#!/usr/bin/env python

# wujian@2019

import yaml
import codecs
import random
import pprint
import pathlib
import argparse

import torch as th
import numpy as np

from asr.utils import StrToBoolAction
from asr.trainer.datp import CtcXentHybridTrainer, TransducerTrainer

from asr.loader import support_loader
from asr.feats import support_transform
from asr.nn import support_nnet

blank_sym = "<blank>"
constrained_conf_keys = [
    "nnet_type", "nnet_conf", "data_conf", "trainer_conf", "asr_transform",
    "enh_transform"
]


def load_conf(yaml_conf, dict_path):
    """
    Load yaml configurations
    """
    # load configurations
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)

    nnet_conf = conf["nnet_conf"]
    # add dictionary info
    with codecs.open(dict_path, encoding="utf-8") as f:
        vocab = {}
        for line in f:
            unit, idx = line.split()
            vocab[unit] = int(idx)

    if "<sos>" not in vocab or "<eos>" not in vocab:
        raise ValueError(f"Missing <sos>/<eos> in {args.dict}")
    nnet_conf["vocab_size"] = len(vocab)

    for key in conf.keys():
        if key not in constrained_conf_keys:
            raise ValueError(f"Invalid configuration item: {key}")
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    trainer_conf = conf["trainer_conf"]
    use_ctc = "ctc_regularization" in trainer_conf and trainer_conf[
        "ctc_regularization"] > 0
    is_transducer = conf["nnet_type"][-10:] == "transducer"
    if not is_transducer:
        nnet_conf["sos"] = vocab["<sos>"]
        nnet_conf["eos"] = vocab["<eos>"]
    # for CTC/RNNT
    if use_ctc or is_transducer:
        if blank_sym not in vocab:
            raise RuntimeError(
                f"Missing {blank_sym} in dictionary for CTC/RNNT training")
        if is_transducer:
            trainer_conf["transducer_blank"] = vocab[blank_sym]
            nnet_conf["blank"] = vocab[blank_sym]
        else:
            trainer_conf["ctc_blank"] = vocab[blank_sym]
            nnet_conf["ctc"] = use_ctc
    return conf, is_transducer


def run(args):
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.random.manual_seed(args.seed)

    # get device
    dev_conf = args.device_ids
    device_ids = tuple(map(int, dev_conf.split(","))) if dev_conf else None

    # new logger instance
    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))),
          flush=True)

    checkpoint = pathlib.Path(args.checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)
    # if exist, resume training
    last_checkpoint = checkpoint / "last.pt.tar"
    resume = args.resume
    if last_checkpoint.exists():
        resume = last_checkpoint.as_posix()

    conf, is_transducer = load_conf(args.conf, args.dict)
    data_conf = conf["data_conf"]
    trn_loader = support_loader(**data_conf["train"],
                                train=True,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                **data_conf["loader"])
    dev_loader = support_loader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                **data_conf["loader"])

    asr_cls = support_nnet(conf["nnet_type"])
    asr_transform = None
    enh_transform = None
    if "asr_transform" in conf:
        asr_transform = support_transform("asr")(**conf["asr_transform"])
    if "enh_transform" in conf:
        enh_transform = support_transform("enh")(**conf["enh_transform"])

    # dump configurations
    with open(checkpoint / "train.yaml", "w") as f:
        yaml.dump(conf, f)

    if enh_transform:
        nnet = asr_cls(enh_transform=enh_transform,
                       asr_transform=asr_transform,
                       **conf["nnet_conf"])
    elif asr_transform:
        nnet = asr_cls(asr_transform=asr_transform, **conf["nnet_conf"])
    else:
        nnet = asr_cls(**conf["nnet_conf"])

    if is_transducer:
        TrainerC = TransducerTrainer
    else:
        TrainerC = CtcXentHybridTrainer
    trainer = TrainerC(nnet,
                       device_ids=device_ids,
                       checkpoint=args.checkpoint,
                       resume=resume,
                       init=args.init,
                       save_interval=args.save_interval,
                       prog_interval=args.prog_interval,
                       tensorboard=args.tensorboard,
                       **conf["trainer_conf"])

    if args.eval_interval > 0:
        trainer.run_batch_per_epoch(trn_loader,
                                    dev_loader,
                                    num_epoches=args.epoches,
                                    eval_interval=args.eval_interval)
    else:
        trainer.run(trn_loader, dev_loader, num_epoches=args.epoches)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ASR model training, configured by yaml files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    parser.add_argument("--device-ids",
                        type=str,
                        default="",
                        help="Training on which GPUs (one or more)")
    parser.add_argument("--epoches",
                        type=int,
                        default=50,
                        help="Number of training epoches")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to save models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--init",
                        type=str,
                        default="",
                        help="Exist model to initialize model training")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of utterances in each batch")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=-1,
                        help="Number of batches trained per epoch "
                        "(for larger training dataset)")
    parser.add_argument("--save-interval",
                        type=int,
                        default=-1,
                        help="Interval to save the checkpoint")
    parser.add_argument("--prog-interval",
                        type=int,
                        default=100,
                        help="Interval to report the progress of the training")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in script data loader")
    parser.add_argument("--tensorboard",
                        action=StrToBoolAction,
                        default="false",
                        help="Flags to use the tensorboad")
    parser.add_argument("--seed",
                        type=int,
                        default=777,
                        help="Random seed used for random package")
    args = parser.parse_args()
    run(args)