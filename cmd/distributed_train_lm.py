#!/usr/bin/env python
# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pprint
import argparse

from aps.utils import set_seed
from aps.conf import load_lm_conf, dump_dict
from aps.opts import DistributedTrainParser
from aps.libs import aps_task, aps_dataloader, aps_asr_nnet, aps_trainer
from aps import distributed


def train_worker(task, conf, vocab_dict, args):
    """
    Initalize training workers
    """
    # init torch/horovod backend
    distributed.init(args.distributed)
    rank = distributed.rank()

    Trainer = aps_trainer(args.trainer, distributed=True)
    trainer = Trainer(task,
                      device_ids=args.device_ids,
                      checkpoint=args.checkpoint,
                      resume=args.resume,
                      save_interval=args.save_interval,
                      prog_interval=args.prog_interval,
                      tensorboard=args.tensorboard,
                      reduction_tag="#tok",
                      **conf["trainer_conf"])
    # dump configurations
    if rank == 0:
        conf["cmd_args"] = vars(args)
        with open(f"{args.checkpoint}/train.yaml", "w") as f:
            yaml.dump(conf, f)
        dump_dict(f"{args.checkpoint}/dict", vocab_dict, reverse=False)

    num_process = len(args.device_ids.split(","))
    if num_process != distributed.world_size():
        raise RuntimeError(f"Number of process != world size: {num_process} " +
                           f"vs {distributed.world_size()}")

    data_conf = conf["data_conf"]
    load_conf = {
        "vocab_dict": vocab_dict,
        "num_workers": args.num_workers // num_process,
        "sos": vocab_dict["<sos>"],
        "eos": vocab_dict["<eos>"],
        "fmt": data_conf["fmt"]
    }
    load_conf.update(data_conf["loader"])
    trn_loader = aps_dataloader(train=True,
                                distributed=True,
                                batch_size=args.batch_size // num_process,
                                **data_conf["train"],
                                **load_conf)
    dev_loader = aps_dataloader(train=False,
                                distributed=False,
                                batch_size=args.batch_size //
                                args.dev_batch_factor,
                                **data_conf["valid"],
                                **load_conf)
    trainer.run(trn_loader,
                dev_loader,
                num_epochs=args.epochs,
                eval_interval=args.eval_interval)


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf, vocab = load_lm_conf(args.conf, args.dict)
    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)

    nnet = aps_asr_nnet(conf["nnet"])(**conf["nnet_conf"])
    task = aps_task(conf["task"], nnet, **conf["task_conf"])

    train_worker(task, conf, vocab, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command for distributed LM model training "
        "Using python -m torch.distributed.launch or horovodrun to launch the command. "
        "See scripts/distributed_train.sh ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DistributedTrainParser.parser])
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    args = parser.parse_args()
    run(args)
