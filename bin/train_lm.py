#!/usr/bin/env python
# wujian@2019

import yaml
import codecs
import random
import pprint
import pathlib
import argparse

import numpy as np
import torch as th

from aps.utils import set_seed
from aps.opts import BaseTrainParser
from aps.trainer.ddp import DdpTrainer
from aps.loader import support_loader
from aps.asr import support_nnet
from aps.task import support_task

constrained_conf_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf"
]


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.full_load(f)

    # create task_conf if None
    if "task_conf" not in conf:
        conf["task_conf"] = {}

    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    # add dictionary info
    with codecs.open(args.dict, encoding="utf-8") as f:
        vocab = {}
        for line in f:
            unit, idx = line.split()
            vocab[unit] = int(idx)

    if "<sos>" not in vocab or "<eos>" not in vocab:
        raise ValueError(f"Missing <sos>/<eos> in {args.dict}")
    eos = vocab["<eos>"]
    sos = vocab["<sos>"]
    conf["nnet_conf"]["vocab_size"] = len(vocab)

    for key in conf.keys():
        if key not in constrained_conf_keys:
            raise ValueError(f"Invalid configuration item: {key}")

    data_conf = conf["data_conf"]
    trn_loader = support_loader(**data_conf["train"],
                                fmt=data_conf["fmt"],
                                train=True,
                                vocab_dict=vocab,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sos=sos,
                                eos=eos)
    dev_loader = support_loader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                vocab_dict=vocab,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sos=sos,
                                eos=eos)

    nnet = support_nnet(conf["nnet"])(**conf["nnet_conf"])
    task = support_task(conf["task"], nnet, **conf["task_conf"])

    trainer = DdpTrainer(task,
                         device_ids=args.device_id,
                         checkpoint=args.checkpoint,
                         resume=args.resume,
                         save_interval=args.save_interval,
                         prog_interval=args.prog_interval,
                         tensorboard=args.tensorboard,
                         **conf["trainer_conf"])
    # dump configurations
    conf["cmd_args"] = vars(args)
    with open(f"{args.checkpoint}/train.yaml", "w") as f:
        yaml.dump(conf, f)

    if args.eval_interval > 0:
        trainer.run_batch_per_epoch(trn_loader,
                                    dev_loader,
                                    num_epochs=args.epochs,
                                    eval_interval=args.eval_interval)
    else:
        trainer.run(trn_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command to start LM training, configured by yaml files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BaseTrainParser.parser])
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    parser.add_argument("--device-id",
                        type=str,
                        default="0",
                        help="Training on which GPU device")
    args = parser.parse_args()
    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))),
          flush=True)
    run(args)
