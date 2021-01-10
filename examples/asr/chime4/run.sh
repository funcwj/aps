#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-4"
chime4_data_dir=/scratch/jwu/CHiME4
wsj0_wav_dir=data/wav/WSJ0

gpu=0
exp=1a
dataset=wsj0_chime4
seed=666
epochs=70
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
  ./local/clean_wsj0_data_prep.sh $chime4_data_dir/CHiME3/data/WSJ0 $wsj0_wav_dir
  ./local/simu_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/real_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/chime4_format_dir.sh
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training AM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --prog-interval 100 \
    --eval-interval -1 \
    am $dataset $exp
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: decoding ..."
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
