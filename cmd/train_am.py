#!/usr/bin/env python

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import yaml
import pprint
import argparse

from aps.utils import set_seed
from aps.conf import load_am_conf, dump_dict
from aps.opts import DistributedTrainParser
from aps.libs import aps_asr_nnet, aps_transform, start_trainer


def run(args):
    # set random seed
    seed = set_seed(args.seed)
    if seed is not None:
        print(f"Set random seed as {seed}")

    conf, vocab = load_am_conf(args.conf, args.dict)

    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)

    asr_cls = aps_asr_nnet(conf["nnet"])
    asr_transform = None
    enh_transform = None
    if "asr_transform" in conf:
        asr_transform = aps_transform("asr")(**conf["asr_transform"])
    if "enh_transform" in conf:
        enh_transform = aps_transform("enh")(**conf["enh_transform"])

    if enh_transform:
        nnet = asr_cls(enh_transform=enh_transform,
                       asr_transform=asr_transform,
                       **conf["nnet_conf"])
    elif asr_transform:
        nnet = asr_cls(asr_transform=asr_transform, **conf["nnet_conf"])
    else:
        nnet = asr_cls(**conf["nnet_conf"])

    other_loader_conf = {
        "vocab_dict": vocab,
    }
    start_trainer(args.trainer,
                  conf,
                  nnet,
                  args,
                  reduction_tag="#tok",
                  other_loader_conf=other_loader_conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command for AM training. To kick off the distributed training, "
        "using python -m torch.distributed.launch "
        "or horovodrun to launch the command. See scripts/distributed_train.sh ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DistributedTrainParser.parser])
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    args = parser.parse_args()
    run(args)
