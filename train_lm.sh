#!/usr/bin/env bash

# wujian@2019

set -eu

epoches=100
tensorboard=false
batch_size=128
chunk_size=20
eval_interval=-1
save_interval=-1

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
dict=data/$data/dict
conf=conf/$data/rnnlm/$exp_id.yaml

[ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

cmd="/home/work_nfs/common/tools/pyqueue_tts.pl"
python=$(which python)

$cmd --gpu 1 $data.train_lm.$exp_id.log \
  $python asr/train_lm.py \
    --conf $conf \
    --dict $dict \
    --tensorboard $tensorboard \
    --checkpoint exp/$data/rnnlm/$exp_id \
    --batch-size $batch_size \
    --chunk-size $chunk_size \
    --epoches $epoches \
    --eval-interval $eval_interval \
    --save-interval $save_interval

mv $data.train_lm.$exp_id.log exp/$data/rnnlm/$exp_id
