#!/usr/bin/env bash

# wujian@2020

set -eu

gpu="0,1"
seed=777
port=10086
epochs=100
distributed="torch"
tensorboard=false
batch_size=128
num_workers=8
num_process=2
eval_interval=4000
save_interval=-1
prog_interval=100

echo "$0 $@"

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2
conf=conf/$data/$exp_id.yaml

[ ! -f $conf ] && echo "$0: missing training configurations $conf" && exit 1

export OMP_NUM_THREADS=4

case $distributed in 
  "torch" )
    python -m torch.distributed.launch \
      --nnodes 1 \
      --nproc_per_node $num_process \
      --master_port $port \
      --use_env \
      bin/distributed_train_ss.py \
      --conf $conf \
      --seed $seed \
      --device-ids $gpu \
      --distributed "torch" \
      --tensorboard $tensorboard \
      --save-interval $save_interval \
      --prog-interval $prog_interval \
      --eval-interval $eval_interval \
      --num-workers $num_workers \
      --checkpoint exp/$data/$exp_id \
      --batch-size $batch_size \
      --epochs $epochs \
      > $data.train_am.$exp_id.log 2>&1
    ;;
  "horovod" )
    horovodrun -np $num_process -H localhost:$num_process \
      python bin/distributed_train_ss.py \
      --conf $conf \
      --seed $seed \
      --device-ids $gpu \
      --distributed "horovod" \
      --tensorboard $tensorboard \
      --save-interval $save_interval \
      --prog-interval $prog_interval \
      --eval-interval $eval_interval \
      --num-workers $num_workers \
      --checkpoint exp/$data/$exp_id \
      --batch-size $batch_size \
      --epochs $epochs \
      > $data.train_ss.$exp_id.log 2>&1
    ;;
  * )
    echo "$0: Unknown --distributed $distributed" && exit 1
    ;;
esac

cp $data.train_am.$exp_id.log exp/$data/$exp_id