#!/usr/bin/env python

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

from aps.utils import set_seed
from aps.opts import DistributedTrainParser
from aps.conf import load_ss_conf
from aps.libs import aps_sse_nnet, aps_transform, start_trainer


def run(args):
    _ = set_seed(args.seed)
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

    is_distributed = args.distributed != "none"
    if is_distributed and args.eval_interval <= 0:
        raise RuntimeError("For distributed training of SE/SS model, "
                           "--eval-interval must be larger than 0")
    start_trainer(args.trainer, conf, nnet, args, reduction_tag="#utt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command for speech separation/enhancement model training. "
        "To kick off the distributed training, using python -m torch.distributed.launch "
        "or horovodrun to launch the command. See scripts/distributed_train.sh ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DistributedTrainParser.parser])
    args = parser.parse_args()
    run(args)
