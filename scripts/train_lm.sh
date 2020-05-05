#!/usr/bin/env bash

# wujian@2019

set -eu

gpu=0
seed=777
epoches=100
tensorboard=false
batch_size=128
eval_interval=-1
save_interval=-1
prog_interval=100

echo "$0 $@"

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
dict=data/$data/dict
conf=conf/$data/rnnlm/$exp_id.yaml

[ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

python src/train_lm.py \
  --conf $conf \
  --dict $dict \
  --seed $seed \
  --device-id $gpu \
  --tensorboard $tensorboard \
  --checkpoint exp/$data/rnnlm/$exp_id \
  --batch-size $batch_size \
  --epoches $epoches \
  --eval-interval $eval_interval \
  --prog-interval $prog_interval \
  --save-interval $save_interval \
  > $data.train_lm.$exp_id.log 2>&1

mv $data.train_lm.$exp_id.log exp/$data/rnnlm/$exp_id