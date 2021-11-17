#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=1
dataset=librimix
librimix_data_dir=/mnt/jwu/librimix

# train
gpu=0
exp=1a
epochs=100
batch_size=32

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: simulate data ..."
  local/simu_librimix.sh $librimix_data_dir
  local/prep_data.sh $librimix_data_dir $dataset
fi

cpt_dir=exp/$dataset/$exp
if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training BSS model ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed 666 \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers 16 \
    --eval-interval -1 \
    --save-interval -1 \
    --prog-interval 250 \
    --tensorboard false \
    ss $dataset $exp
  echo "$0: Train model done under $cpt_dir"
fi
