#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-4"
space="<space>"
dataset=chime4
chime4_data_dir=/home/jwu/doc/data/CHiME4

gpu=0
exp=1a

epochs=60
num_workers=4
batch_size=32

beam_size=16
nbest=8

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  ./local/clean_wsj0_data_prep.sh $chime4_data_dir/CHiME3/data/WSJ0
  ./local/simu_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/real_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/chime4_format_dir.sh
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: tokenizing ..."
  for name in dev train; do
    ./utils/tokenizer.py \
      --dump-vocab data/$dataset/dict \
      --filter-units "<*IN*>,<*MR.*>,<NOISE>" \
      --add-units "<sos>,<eos>,<unk>" \
      --space $space \
      --unit char \
      data/$dataset/$name/text \
      data/$dataset/$name/char
  done
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training AM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed 666 \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --prog-interval 100 \
    --eval-interval -1 \
    am $dataset $exp
fi
