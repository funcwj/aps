#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# data
data=/scratch/jwu/aishell_v1
data_url=www.openslr.org/resources/33

stage=1
dataset="aishell_v1" # prepare data in data/aishell_v1/{train,dev,test}
# training
gpu=0
exp=1a # load training configuration in conf/aishell_v1/1a.yaml
seed=777
epochs=100
tensorboard=false
batch_size=64
num_workers=4
prog_interval=100

# decoding
beam_size=24
nbest=8

. ./utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
  for name in data_aishell resource_aishell; do
    local/download_and_untar.sh $data $data_url $name
  done
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript data/$dataset
fi

if [ $stage -le 2 ]; then
  ./scripts/train_am.sh \
    --seed $seed \
    --gpu $gpu \
    --epochs $epochs \
    --num-workers $num_workers \
    --batch-size $batch_size \
    --tensorboard $tensorboard \
    --prog-interval $prog_interval \
    $dataset $exp
fi

if [ $stage -le 3 ]; then
  # decoding
  ./scripts/decode.sh \
    --gpu $gpu \
    --beam-size $beam_size \
    --nbest $nbest \
    --max-len 50 \
    --dict data/$dataset/dict \
    $dataset $exp \
    data/$dataset/test/wav.scp \
    exp/$dataset/$exp/dec
  # wer
  ./cmd/compute_wer.py \
    exp/$dataset/$exp/dec/beam${beam_size}.decode \
    data/$dataset/test/text
fi
