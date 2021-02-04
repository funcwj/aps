#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

wsj0="/home/jwu/doc/data/wsj0"
wsj1="/home/jwu/doc/data/wsj1"

gpu=0
space="<space>"
stage="1-4"

# am
am_exp=1a
am_seed=888
am_epochs=70
am_batch_size=32
am_num_workers=8

# lm
lm_exp=1a
lm_seed=888
lm_epochs=70
lm_batch_size=32
lm_num_workers=8

# beam search
lm_weight=0.2
beam_size=16
len_norm=true
eos_threshold=0

test_sets="test_dev93 test_eval92"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  ./local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1
  ./local/wsj_format_data.sh || exit 1
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: tokenizing ..."
  for name in test_dev93 train_si284; do
    ./utils/tokenizer.py \
      --dump-vocab data/wsj/dict \
      --filter-units "<*IN*>,<*MR.*>,<NOISE>" \
      --add-units "<sos>,<eos>,<unk>" \
      --space $space \
      --unit char \
      data/wsj/$name/text data/wsj/$name/token
  done
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training AM ..."
  # training am
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $am_seed \
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers $am_num_workers \
    --prog-interval 100 \
    --eval-interval -1 \
    am wsj $am_exp
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: decoding ..."
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true
      --text data/wsj/$name/text \
      --log-suffix $name \
      --beam-size $beam_size \
      --max-len 220 \
      --dict exp/wsj/$am_exp/dict \
      --nbest 8 \
      --space "<space>" \
      --len-norm $len_norm \
      --eos-threshold $eos_threshold \
      wsj $am_exp \
      data/wsj/$name/wav.scp \
      exp/wsj/$am_exp/$name &
  done
  wait
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: preparing data (LM) ..."
  lm_data_dir=data/wsj/lm && mkdir -p $lm_data_dir
  zcat $wsj1/13-32.1/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
    | grep -v "<" | tr "[:lower:]" "[:upper:]" \
    | awk '{printf("utt-%08d %s\n", NR, $0)}' > $lm_data_dir/external.train.text
  ./utils/tokenizer.py \
    --filter-units "<*IN*>,<*MR.*>,<NOISE>" \
    --space $space \
    --unit char \
    $lm_data_dir/external.train.text \
    $lm_data_dir/external.train.token
  cat $lm_data_dir/external.train.token data/wsj/train_si284/token \
    > $lm_data_dir/train.token
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: training LM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $lm_seed \
    --epochs $lm_epochs \
    --batch-size $lm_batch_size \
    --num-workers $lm_num_workers \
    --prog-interval 100 \
    --eval-interval -1 \
    lm wsj $lm_exp
fi

if [ $end -ge 7 ] && [ $beg -le 7 ]; then
  echo "Stage 7: decoding ..."
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true \
      --text data/wsj/$name/text \
      --log-suffix $name \
      --beam-size $beam_size \
      --max-len 220 \
      --lm exp/wsj/nnlm/$lm_exp \
      --lm-weight $lm_weight \
      --len-norm $len_norm \
      --eos-threshold $eos_threshold \
      --dict exp/wsj/$am_exp/dict \
      --nbest 8 \
      --space "<space>" \
      wsj $am_exp \
      data/wsj/$name/wav.scp \
      exp/wsj/$am_exp/${name}_lm${lm_exp}_$lm_weight &
  done
  wait
fi
