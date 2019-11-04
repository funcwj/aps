#!/usr/bin/env python

# wujian@2019

import yaml
import pprint
import argparse

from pathlib import Path
from libs.utils import get_logger, StrToBoolAction
from libs.trainer import S2STrainer
from loader import wav_loader, kaldi_loader
from loader.utils import count_token
from transform.asr import FeatureTransform

from seq2seq import Seq2Seq


def get_dataloader(fmt="wav", **kwargs):
    """
    Got dataloader instance
    """
    if fmt not in ["wav", "kaldi"]:
        raise RuntimeError(f"Unknown data loader: {fmt}")
    func = {
        "wav": wav_loader.make_dataloader,
        "kaldi": kaldi_loader.make_dataloader
    }[fmt]
    return func(**kwargs)


def run(args):
    dev_conf = args.device_ids
    device_ids = tuple(map(int, dev_conf.split(","))) if dev_conf else None

    # new logger instance
    print("Arguments in args:\n{}".format(pprint.pformat(vars(args))),
          flush=True)

    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)
    # if exist, resume training
    last_checkpoint = checkpoint / "last.tar.pt"
    resume = args.resume
    if last_checkpoint.exists():
        resume = last_checkpoint.as_posix()

    # load configurations
    with open(args.conf, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    # add dictionary info
    with open(args.dict, "rb") as f:
        token2idx = [line.decode("utf-8").split()[0] for line in f]

    conf["nnet_conf"]["sos"] = token2idx.index("<sos>")
    conf["nnet_conf"]["eos"] = token2idx.index("<eos>")
    conf["nnet_conf"]["vocab_size"] = len(token2idx)

    print("Arguments in yaml:\n{}".format(pprint.pformat(conf)), flush=True)
    # dump configurations
    with open(checkpoint / "train.yaml", "w") as f:
        yaml.dump(conf, f)

    data_conf = conf["data_conf"]
    trn_loader = get_dataloader(**data_conf["train"],
                                train=True,
                                batch_size=args.batch_size,
                                **data_conf["loader"])
    dev_loader = get_dataloader(**data_conf["valid"],
                                train=False,
                                batch_size=args.batch_size,
                                **data_conf["loader"])
    print(
        f"Number of batches (train/valid) = {len(trn_loader)}/{len(dev_loader)}",
        flush=True)

    if "transform" in conf:
        transform = FeatureTransform(**conf["transform"])
    else:
        transform = None

    trainer_conf = conf["trainer_conf"]
    if trainer_conf["lsm_factor"] > 0:
        token_count = count_token(data_conf["train"]["token_scp"],
                                  conf["nnet_conf"]["vocab_size"])
    else:
        token_count = None
    nnet = Seq2Seq(**conf["nnet_conf"], transform=transform)
    trainer = S2STrainer(nnet,
                         device_ids=device_ids,
                         checkpoint=args.checkpoint,
                         resume=resume,
                         token_count=token_count,
                         save_interval=args.save_interval,
                         tensorboard=args.tensorboard,
                         **trainer_conf)

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
    parser.add_argument("--eval-interval",
                        type=int,
                        default=3000,
                        help="Number of batches trained per epoch "
                        "(for larger training dataset)")
    parser.add_argument("--save-interval",
                        type=int,
                        default=-1,
                        help="Interval to save the checkpoint")
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
