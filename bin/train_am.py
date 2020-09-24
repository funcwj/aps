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
from aps.trainer.ddp import DdpTrainer

from aps.loader import support_loader
from aps.transform import support_transform
from aps.task import support_task
from aps.asr import support_nnet

constrained_conf_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf",
    "asr_transform", "enh_transform"
]


def load_conf(yaml_conf, dict_path):
    """
    Load yaml configurations
    """
    # load configurations
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)
    # create task_conf if None
    if "task_conf" not in conf:
        conf["task_conf"] = {}

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
    task_conf = conf["task_conf"]
    use_ctc = "ctc_weight" in task_conf and task_conf["ctc_weight"] > 0
    is_transducer = conf["task"] == "transducer"
    if not is_transducer:
        nnet_conf["sos"] = vocab["<sos>"]
        nnet_conf["eos"] = vocab["<eos>"]
    # for CTC/RNNT
    if use_ctc or is_transducer:
        conf["task_conf"]["blank"] = len(vocab)
        # add blank
        nnet_conf["vocab_size"] += 1
        if is_transducer:
            nnet_conf["blank"] = len(vocab)
        else:
            nnet_conf["ctc"] = use_ctc
    return conf, vocab


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf, vocab_dict = load_conf(args.conf, args.dict)
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    data_conf = conf["data_conf"]
    trn_loader = support_loader(**data_conf["train"],
                                train=True,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size,
                                vocab_dict=vocab_dict,
                                num_workers=args.num_workers,
                                **data_conf["loader"])
    dev_loader = support_loader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size,
                                vocab_dict=vocab_dict,
                                num_workers=args.num_workers,
                                **data_conf["loader"])

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

    task = support_task(conf["task"], nnet, **conf["task_conf"])

    trainer = DdpTrainer(task,
                         device_ids=args.device_id,
                         checkpoint=args.checkpoint,
                         resume=args.resume,
                         init=args.init,
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
        description=
        "Command to start ASR model training, configured by yaml files",
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
