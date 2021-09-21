#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-4"
space="<space>"
dataset=chime4
track="isolated_1ch_track"

chime4_data_dir=/home/jwu/doc/data/CHiME4
wsj1_data_dir=/home/jwu/doc/data/wsj1

gpu=0
am_exp=1a
lm_exp=1a
am_epochs=60
am_num_workers=32
am_batch_size=128
lm_epochs=100
lm_batch_size=16
lm_num_workers=8

# decoding
beam_size=16
nbest=8
ctc_weight=0.4
lm_weight=0.2
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
  ./local/clean_wsj1_data_prep.sh $wsj1_data_dir
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
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers $am_num_workers \
    --prog-interval 100 \
    --eval-interval -1 \
    am $dataset $am_exp
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: decoding ..."
  for name in $test_sets; do
    name=${name}_${track}
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --beam-size $beam_size \
      --max-len 220 \
      --dict exp/$dataset/$am_exp/dict \
      --nbest $nbest \
      --space "<space>" \
      --ctc-weight $ctc_weight \
      --len-norm $len_norm \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$name &
  done
  wait
fi

lm_data_dir=data/chime4/lm
if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: preparing data (LM) ..."
  mkdir -p $lm_data_dir
  zcat $wsj1_data_dir/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
    | grep -v "<" | tr "[:lower:]" "[:upper:]" \
    | awk '{printf("utt-%08d %s\n", NR, $0)}' > $lm_data_dir/external.train.text
  cat $lm_data_dir/external.train.text \
    data/chime4/train_si200_wsj1_clean/text \
    data/chime4/tr05_orig_clean/text \
    | sort -k1 > $lm_data_dir/train.text
  cat data/chime4/test_dev93_wsj1_clean/text \
    dt05_orig_clean/text | sort -k1 > $lm_data_dir/dev.text
  for name in dev train; do
    ./utils/tokenizer.py \
      --filter-units "<*IN*>,<*MR.*>,<NOISE>" \
      --space $space \
      --unit char \
      $lm_data_dir/$name.text \
      $lm_data_dir/$name.char
  done
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: training LM ..."
  ./scripts/train.sh \
    --gpu 0 \
    --seed 666 \
    --epochs $lm_epochs \
    --batch-size $lm_batch_size \
    --num-workers $lm_num_workers \
    --prog-interval 100 \
    --eval-interval 5000 \
    lm $dataset $lm_exp
fi

if [ $end -ge 7 ] && [ $beg -le 7 ]; then
  echo "Stage 7: decoding ..."
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --beam-size $beam_size \
      --max-len 220 \
      --lm exp/$dataset/nnlm/$lm_exp \
      --lm-weight $lm_weight \
      --len-norm $len_norm \
      --eos-threshold $eos_threshold \
      --dict exp/$dataset/$am_exp/dict \
      --nbest 8 \
      --space "<space>" \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/${name}_lm${lm_exp}_$lm_weight &
  done
  wait
fi
