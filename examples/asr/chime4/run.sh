#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=1
chime4_data_dir=/scratch/jwu/CHiME4
wsj0_wav_dir=data/wav/WSJ0

gpu=0
exp=1a
dataset=wsj0_chime4
num_workers=4
batch_size=32

beam_size=16
nbest=8

. ./utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
  ./local/clean_wsj0_data_prep.sh $chime4_data_dir/CHiME3/data/WSJ0 $wsj0_wav_dir
  ./local/simu_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/real_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/chime4_format_dir.sh
fi

if [ $stage -le 2 ]; then
  ./scripts/train_am.sh \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --gpu $gpu \
    $dataset $exp
fi

if [ $stage -le 3 ]; then
  for name in chime4 wsj0; do
  ./scripts/decode.sh \
    --gpu $gpu \
    --beam-size $beam_size \
    --nbest $nbest \
    --max-len 100 \
    --space "<space>" \
    --dict data/$dataset/dict \
    $dataset $exp \
    data/$name/tst/wav.scp \
    exp/$dataset/$exp/$name
  ./cmd/compute_wer.py \
    exp/$dataset/$exp/$name/beam${beam_size}.decode \
    data/$name/tst/text
  done
fi
