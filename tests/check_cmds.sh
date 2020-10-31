#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

for x in am lm ss; do python ../bin/train_$x.py -h; done
for x in am ss; do python ../bin/distributed_train_$x.py -h; done
for x in wer ss_metric gmvn; do python ../bin/compute_$x.py -h; done
for x in separate_blind decode decode_batch; do python ../bin/$x.py -h; done

# 5.12% & 2.70%
for cer in true false; do
  ../bin/compute_wer.py --cer $cer data/metric/asr/hyp.text \
    data/metric/asr/ref.text
done

for metric in sdr pesq stoi sisnr; do
  ../bin/compute_ss_metric.py --metric $metric \
    data/metric/sse/mix.scp data/metric/sse/ref.scp
done
