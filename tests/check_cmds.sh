#!/usr/bin/env bash

# wujian@2020

set -eu

for x in am lm ss; do python ../bin/train_$x.py -h; done
for x in am ss; do python ../bin/distributed_train_$x.py -h; done
for x in wer ss_metric gmvn; do python ../bin/compute_$x.py -h; done
for x in eval_bss decode decode_batch; do python ../bin/$x.py -h; done
