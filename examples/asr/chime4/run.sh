#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-4"
space="<space>"
dataset=chime4
track="isolated_1ch_track"
chime4_data_dir=/home/jwu/doc/data/CHiME4

gpu=0
am_exp=1a

epochs=60
num_workers=4
batch_size=32

beam_size=16
nbest=8
ctc_weight=0.2
len_norm=true
test_sets="dt05_real dt05_simu et05_real et05_simu"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  ./local/clean_wsj0_data_prep.sh $chime4_data_dir/CHiME3/data/WSJ0
  ./local/simu_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/real_noisy_chime4_data_prep.sh $chime4_data_dir
  ./local/simu_enhan_chime4_data_prep.sh $track $chime4_data_dir/data/audio/16kHz/$track
  ./local/real_enhan_chime4_data_prep.sh $track $chime4_data_dir/data/audio/16kHz/$track
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

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: decoding ..."
  for name in $test_sets; do
    name=${name}_${$track}
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --beam-size $beam_size \
      --max-len 220 \
      --dict exp/$dataset/$am_exp/dict \
      --nbest 8 \
      --space "<space>" \
      --ctc-weight $ctc_weight \
      --len-norm $len_norm \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$name &
  done
  wait
fi
