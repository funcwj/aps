#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pprint
import argparse

from aps.utils import set_seed
from aps.opts import BaseTrainParser
from aps.conf import load_ss_conf
from aps.libs import aps_transform, aps_task, aps_dataloader, aps_sse_nnet, aps_trainer


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf = load_ss_conf(args.conf)
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)

    data_conf = conf["data_conf"]
    load_conf = {
        "fmt": data_conf["fmt"],
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    load_conf.update(data_conf["loader"])
    trn_loader = aps_dataloader(train=True, **load_conf, **data_conf["train"])
    dev_loader = aps_dataloader(train=False, **load_conf, **data_conf["valid"])

    sse_cls = aps_sse_nnet(conf["nnet"])
    if "enh_transform" in conf:
        enh_transform = aps_transform("enh")(**conf["enh_transform"])
        nnet = sse_cls(enh_transform=enh_transform, **conf["nnet_conf"])
    else:
        nnet = sse_cls(**conf["nnet_conf"])

    task = aps_task(conf["task"], nnet, **conf["task_conf"])

    Trainer = aps_trainer(args.trainer, distributed=False)
    trainer = Trainer(task,
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

    trainer.run(trn_loader,
                dev_loader,
                num_epochs=args.epochs,
                eval_interval=args.eval_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command for speech separation/enhancement model training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[BaseTrainParser.parser])
    parser.add_argument("--device-id",
                        type=str,
                        default="0",
                        help="Training on which GPU device")
    args = parser.parse_args()
    run(args)
