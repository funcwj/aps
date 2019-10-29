#!/usr/bin/env python

# wujian@2019

import pprint
import argparse
import yaml

from pathlib import Path
from libs.logger import get_logger
from libs.trainer import S2STrainer
from libs.dataset import make_dataloader

from seq2seq import Seq2Seq


def run(args):
    dev_conf = args.device_ids
    device_ids = tuple(map(int, dev_conf.split(","))) if dev_conf else None

    # new logger instance
    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))),
          flush=True)

    checkpoint = Path(args.checkpoint)
    # if exist, resume training
    last_checkpoint = checkpoint / "last.tar.pt"
    resume = args.resume
    if last_checkpoint.exists():
        resume = last_checkpoint.as_posix()

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)

    # add dictionary info
    with open(args.dict, "rb") as f:
        token2idx = [line.decode("utf-8").split()[0] for line in f]
        conf["nnet_conf"]["sos"] = token2idx.index("<sos>")
        conf["nnet_conf"]["eos"] = token2idx.index("<eos>")
        conf["nnet_conf"]["vocab_size"] = len(token2idx)

    # dump configurations
    with open(checkpoint / "train.yaml", "w") as f:
        yaml.dump(conf, f)

    nnet = Seq2Seq(**conf["nnet_conf"])
    trainer = S2STrainer(nnet,
                         device_ids=device_ids,
                         checkpoint=args.checkpoint,
                         resume=resume,
                         **conf["trainer_conf"])
    data_conf = conf["data_conf"]
    train_loader = make_dataloader(**data_conf["train"],
                                   train=True,
                                   batch_size=args.batch_size,
                                   **data_conf["loader"])
    valid_loader = make_dataloader(**data_conf["valid"],
                                   train=False,
                                   batch_size=args.batch_size,
                                   **data_conf["loader"])

    if args.eval_interval > 0:
        trainer.run_batch_per_epoch(train_loader,
                                    valid_loader,
                                    num_epoches=args.epoches,
                                    eval_interval=args.eval_interval)
    else:
        trainer.run(train_loader, valid_loader, num_epoches=args.epoches)


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
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
                        help="Number of batches trained per epoch "
                        "(for larger training dataset)")
    args = parser.parse_args()
    run(args)
