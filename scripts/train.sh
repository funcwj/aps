#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# wrapper for train_{ss,am,lm}.py
set -eu

gpu=0
seed=777
epochs=100
trainer="ddp"
tensorboard=false
batch_size=64
num_workers=4
eval_interval=-1
save_interval=-1
prog_interval=100

echo "$0 $*"

. ./utils/parse_options.sh || exit 1

[ $# -ne 3 ] && echo "Script format error: $0 <task> <data-set> <exp-id>" && exit 1

task=$1
data=$2
exp_id=$3

conf=conf/$data/$exp_id.yaml
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

opts=""
case $task in
  "am" | "lm" )
    dict=data/$data/dict
    [ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
    opts="--dict $dict"
    ;;
  "ss" )
    ;;
  * )
    echo "$0: Unknown task: $task" && exit 1
    ;;
esac

cmd=train_$task

cmd/$cmd.py $opts \
  --conf $conf \
  --seed $seed \
  --epochs $epochs \
  --trainer $trainer \
  --device-id $gpu \
  --batch-size $batch_size \
  --checkpoint exp/$data/$exp_id \
  --num-workers $num_workers \
  --tensorboard $tensorboard \
  --save-interval $save_interval \
  --prog-interval $prog_interval \
  --eval-interval $eval_interval \
  > $data.$cmd.$exp_id.log 2>&1

cp $data.$cmd.$exp_id.log exp/$data/$exp_id
