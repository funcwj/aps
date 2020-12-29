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
am_exp=1a # load training configuration in conf/aishell_v1/1a.yaml
lm_exp=1a # load training configuration in conf/aishell_v1/nnlm/1a.yaml

seed=777
tensorboard=false
prog_interval=100

# for am
am_epochs=100
am_batch_size=64
am_num_workers=4

# for lm
lm_epochs=30
lm_batch_size=32
lm_num_workers=4

# decoding
beam_size=24
nbest=8
lm_weight=0.2

. ./utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
  for name in data_aishell resource_aishell; do
    local/download_and_untar.sh $data $data_url $name
  done
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript data/$dataset
fi

if [ $stage -le 2 ]; then
  ./scripts/train.sh \
    --seed $seed \
    --gpu $gpu \
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers $am_num_workers \
    --tensorboard $tensorboard \
    --prog-interval $prog_interval \
    am $dataset $am_exp
fi

if [ $stage -le 3 ]; then
  # decoding
  ./scripts/decode.sh \
    --gpu $gpu \
    --beam-size $beam_size \
    --nbest $nbest \
    --max-len 50 \
    --dict data/$dataset/dict \
    $dataset $am_exp \
    data/$dataset/test/wav.scp \
    exp/$dataset/$am_exp/dec
  # wer
  ./cmd/compute_wer.py \
    exp/$dataset/$am_exp/dec/beam${beam_size}.decode \
    data/$dataset/test/text
fi

if [ $stage -le 4 ]; then
  ./scripts/train.sh \
    --seed $seed \
    --gpu $gpu \
    --epochs $lm_epochs \
    --batch-size $lm_batch_size \
    --num-workers $lm_num_workers \
    --tensorboard $tensorboard \
    --prog-interval $prog_interval \
    lm $dataset $lm_exp
fi

if [ $stage -le 5 ]; then
  name=dec_lm${lm_exp}_$lm_weight
  # decoding
  ./scripts/decode.sh \
    --lm exp/aishell_v1/nnlm/$lm_exp \
    --gpu $gpu \
    --dict data/$dataset/dict \
    --nbest $nbest \
    --max-len 50 \
    --beam-size $beam_size \
    --lm-weight $lm_weight \
    $dataset $am_exp \
    data/$dataset/test/wav.scp \
    exp/$dataset/$am_exp/$name
  # wer
  ./cmd/compute_wer.py \
    exp/$dataset/$am_exp/$name/beam${beam_size}.decode \
    data/$dataset/test/text
fi
