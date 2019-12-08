#!/usr/bin/env python
# wujian@2019

import yaml
import pprint
import pathlib
import argparse

from libs.utils import StrToBoolAction
from libs.trainer import LmTrainer
from loader import support_loader
from nn import support_nnet

constrained_conf_keys = ["nnet_type", "nnet_conf", "data_conf", "trainer_conf"]


def run(args):
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

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.full_load(f)

    # add dictionary info
    with open(args.dict, "rb") as f:
        token2idx = [line.decode("utf-8").split()[0] for line in f]

    eos = token2idx.index("<eos>")
    conf["nnet_conf"]["vocab_size"] = len(token2idx)
    conf["nnet_type"] = "rnnlm"

    for key in conf.keys():
        if key not in constrained_conf_keys:
            raise ValueError(f"Invalid configuration item: {key}")

    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    # dump configurations
    with open(checkpoint / "train.yaml", "w") as f:
        yaml.dump(conf, f)

    data_conf = conf["data_conf"]
    trn_loader = support_loader(**data_conf["train"],
                                fmt=data_conf["fmt"],
                                train=True,
                                batch_size=args.batch_size,
                                chunk_size=args.chunk_size,
                                eos=eos)
    dev_loader = support_loader(**data_conf["valid"],
                                train=False,
                                fmt=data_conf["fmt"],
                                batch_size=args.batch_size,
                                chunk_size=args.chunk_size,
                                eos=eos)
    print("Number of batches (train/valid) = " +
          f"{len(trn_loader)}/{len(dev_loader)}",
          flush=True)

    asr_cls = support_nnet(conf["nnet_type"])
    nnet = asr_cls(**conf["nnet_conf"])

    trainer = LmTrainer(nnet,
                        device_ids=device_ids,
                        checkpoint=args.checkpoint,
                        resume=resume,
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
        description="Command to start model training, configured by yaml files",
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
    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
                        help="Number of utterances in each batch")
    parser.add_argument("--chunk-size",
                        type=int,
                        default=32,
                        help="Bptt size used during training")
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
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
    args = parser.parse_args()
    run(args)
