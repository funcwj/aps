#!/usr/bin/env bash

# wujian@2019

set -eu

node=""
epoches=100
tensorboard=false
batch_size=64
num_workers=4
eval_interval=-1
save_interval=-1
prog_interval=100

echo "$0 $@"

. ./local/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
dict=data/$data/dict
conf=conf/$data/$exp_id.yaml

[ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

cmd="/home/work_nfs/common/tools/pyqueue_asr.pl"
opts="--gpu 1"
python=$(which python)

[ ! -z $node ] && opts="$opts -l hostname=$node"

$cmd $opts $data.train_am.$exp_id.log \
  $python asr/train_am.py \
    --conf $conf \
    --dict $dict \
    --tensorboard $tensorboard \
    --save-interval $save_interval \
    --prog-interval $prog_interval \
    --num-workers $num_workers \
    --checkpoint exp/$data/$exp_id \
    --batch-size $batch_size \
    --epoches $epoches \
    --eval-interval $eval_interval

mv $data.train_am.$exp_id.log exp/$data/$exp_id