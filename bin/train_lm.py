#!/usr/bin/env python
# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pprint
import argparse

from aps.utils import set_seed
from aps.opts import BaseTrainParser
from aps.conf import load_lm_conf
from aps.libs import aps_task, aps_dataloader, aps_asr_nnet, aps_trainer


def run(args):
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf, vocab = load_lm_conf(args.conf, args.dict)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)
    eos = vocab["<eos>"]
    sos = vocab["<sos>"]
    conf["nnet_conf"]["vocab_size"] = len(vocab)

    data_conf = conf["data_conf"]
    trn_loader = aps_dataloader(**data_conf["train"],
                                fmt=data_conf["fmt"],
                                train=True,
                                vocab_dict=vocab,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sos=sos,
                                eos=eos)
    dev_loader = aps_dataloader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                vocab_dict=vocab,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                sos=sos,
                                eos=eos)

    nnet = aps_asr_nnet(conf["nnet"])(**conf["nnet_conf"])
    task = aps_task(conf["task"], nnet, **conf["task_conf"])

    Trainer = aps_trainer(args.trainer, distributed=False)
    trainer = Trainer(task,
                      device_ids=args.device_id,
                      checkpoint=args.checkpoint,
                      resume=args.resume,
                      save_interval=args.save_interval,
                      prog_interval=args.prog_interval,
                      tensorboard=args.tensorboard,
                      opt_level=args.opt_level,
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
    run(args)
