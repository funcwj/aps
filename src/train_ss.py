#!/usr/bin/env python

# wujian@2020

import yaml
import codecs
import random
import pprint
import pathlib
import argparse

import torch as th
import numpy as np

from aps.utils import StrToBoolAction
from aps.trainer.ddp import Trainer

from aps.loader import support_loader
from aps.feats import support_transform
from aps.task import support_task
from aps.ss import support_nnet

constrained_conf_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf",
    "enh_transform"
]


def load_conf(yaml_conf):
    """
    Load yaml configurations
    """
    # load configurations
    with open(yaml_conf, "r") as f:
        conf = yaml.full_load(f)

    # create task_conf if None
    if "task_conf" not in conf:
        conf["task_conf"] = {}

    for key in conf.keys():
        if key not in constrained_conf_keys:
            raise ValueError(f"Invalid configuration item: {key}")

    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    return conf


def run(args):
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.random.manual_seed(args.seed)

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

    conf = load_conf(args.conf)
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

    asr_cls = support_nnet(conf["nnet"])
    if "enh_transform" in conf:
        enh_transform = support_transform("enh")(**conf["enh_transform"])
        nnet = asr_cls(enh_transform=enh_transform, **conf["nnet_conf"])
    else:
        nnet = asr_cls(**conf["nnet_conf"])

    # dump configurations
    with open(checkpoint / "train.yaml", "w") as f:
        yaml.dump(conf, f)

    task = support_task(conf["task"], nnet, **conf["task_conf"])

    trainer = Trainer(task,
                      device_ids=args.device_id,
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
        description="Command for speech separation/enhancement model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf",
                        type=str,
                        required=True,
                        help="Yaml configuration file for training")
    parser.add_argument("--device-id",
                        type=int,
                        default=0,
                        help="Training on which GPU device")
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
