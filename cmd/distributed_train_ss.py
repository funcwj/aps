#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pprint
import argparse

from aps.utils import set_seed
from aps.opts import DistributedTrainParser
from aps.conf import load_ss_conf
from aps.libs import aps_transform, aps_task, aps_dataloader, aps_sse_nnet, aps_trainer
from aps import distributed


def train_worker(task, conf, args):
    """
    Initalize training workers
    """
    # init torch/horovod backend
    distributed.init(args.distributed)
    rank = distributed.rank()

    Trainer = aps_trainer(args.trainer, distributed=True)
    trainer = Trainer(task,
                      rank=distributed.rank(),
                      device_ids=args.device_ids,
                      checkpoint=args.checkpoint,
                      resume=args.resume,
                      init=args.init,
                      save_interval=args.save_interval,
                      prog_interval=args.prog_interval,
                      tensorboard=args.tensorboard,
                      **conf["trainer_conf"])

    # dump configurations
    if rank == 0:
        conf["cmd_args"] = vars(args)
        with open(f"{args.checkpoint}/train.yaml", "w") as f:
            yaml.dump(conf, f)

    num_process = len(args.device_ids.split(","))
    if num_process != distributed.world_size():
        raise RuntimeError(f"Number of process != world size: {num_process} " +
                           f"vs {distributed.world_size()}")
    data_conf = conf["data_conf"]
    load_conf = {
        "fmt": data_conf["fmt"],
        "num_workers": args.num_workers // num_process
    }
    load_conf.update(data_conf["loader"])
    trn_loader = aps_dataloader(train=True,
                                distributed=True,
                                batch_size=args.batch_size // num_process,
                                **load_conf,
                                **data_conf["train"])
    dev_loader = aps_dataloader(train=False,
                                distributed=False,
                                batch_size=args.batch_size //
                                args.dev_batch_factor,
                                **load_conf,
                                **data_conf["valid"])
    if args.eval_interval <= 0:
        raise RuntimeError("For distributed training of SE/SS model, "
                           "--eval-interval must be larger than 0")
    trainer.run(trn_loader,
                dev_loader,
                num_epochs=args.epochs,
                eval_interval=args.eval_interval)


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf = load_ss_conf(args.conf)
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)

    sse_cls = aps_sse_nnet(conf["nnet"])
    # with or without enh_tranform
    if "enh_transform" in conf:
        enh_transform = aps_transform("enh")(**conf["enh_transform"])
        nnet = sse_cls(enh_transform=enh_transform, **conf["nnet_conf"])
    else:
        nnet = sse_cls(**conf["nnet_conf"])

    task = aps_task(conf["task"], nnet, **conf["task_conf"])
    train_worker(task, conf, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command for distributed speech separation/enhancement model training "
        "Using python -m torch.distributed.launch or horovodrun to launch the command. "
        "See scripts/distributed_train.sh ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DistributedTrainParser.parser])
    args = parser.parse_args()
    run(args)
