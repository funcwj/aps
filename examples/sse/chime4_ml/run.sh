#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=1
chime4_data=/scratch/jwu/CHiME4
cache_dir=data/chime4_wav
dataset="chime4_ml"

# training
exp="1a"
gpu=0
epochs=100
batch_size=32
num_workers=4

. ./utils/parse_options.sh || exit 1

[ $stage -le 1 ] && local/prep_data.sh --dataset $dataset $chime4_data $cache_dir

if [ $stage -le 2 ]; then
  ./scripts/train.sh \
    --gpu $gpu \
    --seed 777 \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --eval-interval -1 \
    --save-interval -1 \
    --prog-interval 50 \
    --tensorboard false \
    ss $dataset $exp
  echo "$0: Train model done under exp/$dataset/$exp"
fi

if [ $stage -le 3 ]; then
  ./local/eval.py \
    --checkpoint exp/$dataset/$exp \
    --sr 16000 \
    data/$dataset/tst.scp exp/$dataset/$exp/mask
fi
