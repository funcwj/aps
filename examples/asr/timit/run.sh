#!/usr/bin/env bash

# wujian@2020

# 1a.yaml
#   WER(%) Report:
#   Total WER: 26.13%, 192 utterances

set -eu

timit_data=/scratch/jwu/TIMIT-LDC93S1/TIMIT
dataset="timit"
stage=1
# training
gpu=0
exp=1a # load training configuration in conf/timit/1a.yaml
seed=777
epochs=100
tensorboard=false
batch_size=8
num_workers=2
prog_interval=100

# decoding
beam_size=32
nbest=8

. ./utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
  ./local/timit_data_prep.sh --dataset $dataset $timit_data
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
    "timit" $exp
fi

if [ $stage -le 3 ]; then
  # decoding
  ./scripts/decode.sh \
    --gpu $gpu \
    --beam-size $beam_size \
    --nbest $nbest \
    --max-len 75 \
    --dict data/timit/dict \
    "timit" $exp \
    data/timit/test/wav.scp \
    exp/timit/$exp/dec
  # wer
  ./bin/compute_wer.py exp/timit/$exp/dec/beam24.decode data/timit/test/text
fi
