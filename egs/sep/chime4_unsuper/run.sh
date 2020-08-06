#!/usr/bin/env bash

# wujian@2019

set -eu 

stage=1
chime4_data=/scratch/jwu/CHiME4
cache_dir=data/chime4_wav

dataset="chime4_unsuper"
exp="1a"
gpu=0
seed=777
epochs=100
tensorboard=false
batch_size=8
num_workers=4
eval_interval=-1
save_interval=-1
prog_interval=100

. ./utils/parse_options.sh || exit 1

[ $stage -le 1 ] && local/prep_data.sh --dataset $dataset $chime4_data $cache_dir || exit 1

if [ $stage -le 2]; then
    ./scripts/train_ss.sh \
        --gpu $gpu --seed $seed \
        --epochs $epochs --batch-size $batch_size \
        --num-workers $num_workers \
        --eval-interval $eval_interval \
        --save-interval $save_interval \
        --prog-interval $prog_interval \
        --tensorboard $tensorboard \
        $dataset $exp
    echo "$0: Train model done under exp/$dataset/$exp"
fi