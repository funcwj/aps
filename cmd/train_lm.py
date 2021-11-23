#!/usr/bin/env python
# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import pprint
import argparse

from aps.utils import set_seed
from aps.const import SOS_TOKEN, EOS_TOKEN
from aps.conf import load_lm_conf, dump_dict
from aps.opts import DistributedTrainParser
from aps.libs import aps_asr_nnet, start_trainer


def run(args):
    _ = set_seed(args.seed)
    conf, vocab = load_lm_conf(args.conf, args.dict)

    print(f"Arguments in args:\n{pprint.pformat(vars(args))}", flush=True)
    print(f"Arguments in yaml:\n{pprint.pformat(conf)}", flush=True)

    nnet = aps_asr_nnet(conf["nnet"])(**conf["nnet_conf"])

    other_loader_conf = {
        "vocab_dict": vocab,
        "sos": vocab[SOS_TOKEN],
        "eos": vocab[EOS_TOKEN],
    }
    start_trainer(args.trainer,
                  conf,
                  nnet,
                  args,
                  reduction_tag="#tok",
                  other_loader_conf=other_loader_conf)
    dump_dict(f"{args.checkpoint}/dict", vocab, reverse=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to train language models (LM). To kick off the distributed training, "
        "using torchrun (python -m torch.distributed.launch) "
        "or horovodrun to launch the command. See scripts/distributed_train.sh ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[DistributedTrainParser.parser])
    parser.add_argument("--dict",
                        type=str,
                        required=True,
                        help="Dictionary file")
    args = parser.parse_args()
    run(args)
