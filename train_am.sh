#!/usr/bin/env bash

# wujian@2019

set -eu

epoches=100
tensorboard=false
batch_size=64
num_workers=4
eval_interval=-1
save_interval=-1

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
dict=data/$data/dict
conf=conf/$data/$exp_id.yaml

[ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

cmd="/home/work_nfs/common/tools/pyqueue_tts.pl"
python=$(which python)

$cmd --gpu 1 -l hostname=node4 $data.train_am.$exp_id.log \
  $python asr/train_am.py \
    --conf $conf \
    --dict $dict \
    --tensorboard $tensorboard \
    --save-interval $save_interval \
    --num-workers $num_workers \
    --checkpoint exp/$data/$exp_id \
    --batch-size $batch_size \
    --epoches $epoches \
    --eval-interval $eval_interval

mv $data.train.$exp_id.log exp/$data/$exp_id
