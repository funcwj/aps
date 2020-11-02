#!/usr/bin/env python
# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import codecs
import pprint
import argparse

from aps.utils import set_seed
from aps.opts import BaseTrainParser
from aps.trainer.ddp import DdpTrainer
from aps.loader import support_loader
from aps.task import support_task
from aps.conf import load_lm_conf
from aps.asr import support_nnet

constrained_conf_keys = [
    "nnet", "nnet_conf", "task", "task_conf", "data_conf", "trainer_conf"
]


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf, vocab = load_lm_conf(args.conf, args.dict)
    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    eos = vocab["<eos>"]
    sos = vocab["<sos>"]
    conf["nnet_conf"]["vocab_size"] = len(vocab)

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
