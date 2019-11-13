#!/usr/bin/env bash

# wujian@2019

set -eu

dict=data/aishell_v1/dict
epoches=100
batch_size=96
num_workers=4
eval_interval=-1
save_interval=-1

echo "$0 $@"

[ $# -ne 1 ] && echo "Script format error: $0 <exp-id>" && exit 1

exp_id=$1

cmd="/home/work_nfs/common/tools/pyqueue_asr.pl"
python=$(which python)

$cmd --gpu 1 train.$exp_id.log \
  $python asr/train_am.py \
    --conf conf/$exp_id.yaml \
    --dict $dict \
    --tensorboard false \
    --save-interval $save_interval \
    --num-workers $num_workers \
    --checkpoint exp/aishell_v1/$exp_id \
    --batch-size $batch_size \
    --epoches $epoches \
    --eval-interval $eval_interval

mv train.$exp_id.log exp/aishell_v1/$exp_id
