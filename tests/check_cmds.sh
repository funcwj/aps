#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

for x in am lm ss; do python ../bin/train_$x.py -h; done
for x in am ss; do python ../bin/distributed_train_$x.py -h; done
for x in wer ss_metric gmvn; do python ../bin/compute_$x.py -h; done
for x in separate_blind decode decode_batch; do python ../bin/$x.py -h; done
