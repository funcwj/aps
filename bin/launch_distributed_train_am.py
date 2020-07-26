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

from aps.utils import set_seed
from aps.opts import BaseTrainParser
from aps.trainer.ddp import Trainer

from aps.loader import support_loader
from aps.transform import support_transform
from aps.task import support_task
from aps.asr import support_nnet

blank_sym = "<blank>"
constrained_conf_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf",
    "asr_transform", "enh_transform"
]


def train_worker(rank, nnet, conf, args):
    """
    Initalize training workers
    """
    # construct trainer
    # torch.distributed.launch will provide
    # environment variables, and requires that you use init_method="env://".

    task = support_task(conf["task"], nnet, **conf["task_conf"])

    trainer = Trainer(task,
                      rank=rank,
                      device_ids=tuple(i for i in range(args.num_process)),
                      checkpoint=args.checkpoint,
                      resume=args.resume,
                      init=args.init,
                      save_interval=args.save_interval,
                      prog_interval=args.prog_interval,
                      tensorboard=args.tensorboard,
                      **conf["trainer_conf"])

    data_conf = conf["data_conf"]
    trn_loader = support_loader(**data_conf["train"],
                                train=True,
                                distributed=True,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size // args.num_process,
                                num_workers=args.num_workers,
                                **data_conf["loader"])
    dev_loader = support_loader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size // args.num_process,
                                num_workers=args.num_workers,
                                **data_conf["loader"])
    if args.eval_interval > 0:
        trainer.run_batch_per_epoch(trn_loader,
                                    dev_loader,
                                    num_epochs=args.epochs,
                                    eval_interval=args.eval_interval)
    else:
        trainer.run(trn_loader, dev_loader, num_epochs=args.epochs)


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

    nnet_conf["vocab_size"] = len(vocab)

    for key in conf.keys():
        if key not in constrained_conf_keys:
            raise ValueError(f"Invalid configuration item: {key}")
    task_conf = conf["task_conf"]
    use_ctc = "ctc_weight" in task_conf and task_conf["ctc_weight"] > 0
    is_transducer = conf["task"] == "transducer"
    if not is_transducer:
        nnet_conf["sos"] = vocab["<sos>"]
        nnet_conf["eos"] = vocab["<eos>"]
    # for CTC/RNNT
    if use_ctc or is_transducer:
        if blank_sym not in vocab:
            raise RuntimeError(
                f"Missing {blank_sym} in dictionary for CTC/RNNT training")
        conf["task_conf"]["blank"] = vocab[blank_sym]
        if is_transducer:
            nnet_conf["blank"] = vocab[blank_sym]
        else:
            nnet_conf["ctc"] = use_ctc
    return conf


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    if th.cuda.device_count() < args.num_process:
        raise RuntimeError("--num-process exceeds number of the GPUs")

    conf = load_conf(args.conf, args.dict)
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)

    asr_cls = support_nnet(conf["nnet"])
    asr_transform = None
    enh_transform = None
    if "asr_transform" in conf:
        asr_transform = support_transform("asr")(**conf["asr_transform"])
    if "enh_transform" in conf:
        enh_transform = support_transform("enh")(**conf["enh_transform"])

    if enh_transform:
        nnet = asr_cls(enh_transform=enh_transform,
                       asr_transform=asr_transform,
                       **conf["nnet_conf"])
    elif asr_transform:
        nnet = asr_cls(asr_transform=asr_transform, **conf["nnet_conf"])
    else:
        nnet = asr_cls(**conf["nnet_conf"])

    train_worker(args.local_rank, nnet, conf, args)

    # dump configurations
    with open(f"{args.checkpoint}/train.yaml", "w") as f:
        yaml.dump(conf, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ASR model training, configured by yaml files "
        "(support distributed mode on single node). "
        "Using python -m torch.distributed.launch to launch the command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BaseTrainParser.parser])
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help="Local rank value, supplied automatically "
                        "by torch.distributed.launch")
    parser.add_argument("--num-process",
                        type=int,
                        default=2,
                        help="Number of process for distributed training")
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    args = parser.parse_args()
    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))),
          flush=True)
    run(args)
