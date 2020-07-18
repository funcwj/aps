#!/usr/bin/env bash

# wujian@2019

set -eu

seed=777
port=10086
epochs=100
tensorboard=false
batch_size=64
num_workers=8
num_process=2
eval_interval=-1
save_interval=-1
prog_interval=100

echo "$0 $@"

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
dict=data/$data/dict
conf=conf/$data/$exp_id.yaml

[ ! -f $dict ] && echo "$0: missing dictionary $dict" && exit 1
[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

export OMP_NUM_THREADS=24

python -m torch.distributed.launch \
  --nnodes=1 \
  --nproc_per_node=$num_process \
  --master_port=$port \
  bin/launch_distributed_train_am.py \
  --conf $conf \
  --dict $dict \
  --seed $seed \
  --tensorboard $tensorboard \
  --save-interval $save_interval \
  --prog-interval $prog_interval \
  --num-workers $num_workers \
  --num-process $num_process \
  --checkpoint exp/$data/$exp_id \
  --batch-size $batch_size \
  --epochs $epochs \
  --eval-interval $eval_interval \
  > $data.train_am.$exp_id.log 2>&1

cp $data.train_am.$exp_id.log exp/$data/$exp_id