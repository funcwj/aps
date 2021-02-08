#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-7"
data=/scratch/jwu/LibriSpeech

# word piece
wp_name="wpm_16k"
wp_mode="unigram"
vocab_size=16000


# am
am_exp=1a
am_seed=888
am_batch_size=96
am_num_workers=16
am_epochs=70
am_eval_interval=3000
am_prog_interval=100

# lm
lm_exp=1a
lm_seed=777
lm_batch_size=32
lm_eval_interval=20000
lm_prog_interval=100
lm_epochs=30
lm_num_workers=4

# decoding
nbest=16
beam_size=8
len_norm=true
lm_weight=0.2
eos_threshold=1
test_sets="test_clean test_other"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

# NOTE: download Librispeech data first
if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data for AM ..."
  for x in dev-{clean,other} test-{clean,other} train-clean-{100,360} train-other-500; do
    ./local/data_prep.sh $data/$x data/librispeech/"$(echo $x | sed s/-/_/g)"
  done
  ./local/format_data.sh --nj 10 data/librispeech
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: tokenizing ..."
  # training
  ./utils/subword.sh --op "train" --mode $wp_mode --vocab-size $vocab_size \
    data/librispeech/train/text exp/librispeech/$wp_name
  cp exp/librispeech/$wp_name/dict data/librispeech
  # wp encoding
  for data in dev train; do
    ./utils/subword.sh --op "encode" --encode "piece" \
      data/librispeech/$data/text exp/librispeech/$wp_name \
      > data/librispeech/$data/token
  done
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training AM ..."
  # training am
  ./scripts/distributed_train.sh \
    --seed $am_seed \
    --gpu "0,1,2,3" \
    --num-process 4 \
    --distributed "torch" \
    --epochs $am_epochs \
    --num-workers $am_num_workers \
    --batch-size $am_batch_size \
    --prog-interval $am_prog_interval \
    --eval-interval $am_eval_interval \
    am librispeech $am_exp
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: decoding ..."
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true \
      --text data/librispeech/$name/text \
      --log-suffix $name \
      --beam-size $beam_size \
      --max-len 150 \
      --len-norm $len_norm \
      --dict exp/librispeech/$am_exp/dict \
      --nbest $nbest \
      --spm exp/librispeech/$wp_name/$wp_mode.model \
      librispeech $am_exp \
      data/librispeech/$name/wav.scp \
      exp/librispeech/$am_exp/$name &
  done
  wait
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: preparing data for LM ..."
  # prepare lm data
  [ ! -f $data/lm/librispeech-lm-norm.txt ] && \
  mkdir $data/lm && cd $data/lm && wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz && \
    gunzip librispeech-lm-norm.txt.gz && cd -
  lm_data_dir=data/librispeech/lm && mkdir -p $lm_data_dir
  cat data/librispeech/dev/token | sort -k1 > $lm_data_dir/dev.token
  cp $data/lm/librispeech-lm-norm.txt $lm_data_dir/train.text
  ./utils/subword.sh --op "encode" --encode "piece" \
    $lm_data_dir/train.text exp/librispeech/$wp_name \
    > $lm_data_dir/external.train.token
  awk '{$1=""; print}' data/librispeech/train/token | sed 's:^[ \t]*::g' \
    > $lm_data_dir/am.train.token
  cat $lm_data_dir/{external,am}.train.token > $lm_data_dir/train.token
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: training LM ..."
  # training lm
  ./scripts/train.sh \
    --seed $lm_seed \
    --epochs $lm_epochs \
    --batch-size $lm_batch_size \
    --num-workers $lm_num_workers \
    --prog-interval $lm_prog_interval \
    --eval-interval $lm_eval_interval \
    lm librispeech $lm_exp
fi

if [ $end -ge 7 ] && [ $beg -le 7 ]; then
  echo "Stage 7: decoding ..."
  # shallow fusion
  for name in $test_sets; do
    dec_name=${name}_lm${lm_exp}_$lm_weight
    ./scripts/decode.sh \
      --score true \
      --text data/librispeech/$name/text \
      --log-suffix $name \
      --spm exp/librispeech/$wp_name/$wp_mode.model \
      --lm exp/librispeech/nnlm/$lm_exp \
      --lm-weight $lm_weight \
      --beam-size $beam_size \
      --max-len 150 \
      --len-norm $len_norm \
      --eos-threshold $eos_threshold \
      --dict exp/librispeech/$am_exp/dict \
      --nbest $nbest \
      librispeech $am_exp \
      data/librispeech/$name/wav.scp \
      exp/librispeech/$am_exp/$dec_name &
  done
  wait
fi
