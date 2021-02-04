#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# data
data=/scratch/jwu/aishell_v1
data_url=www.openslr.org/resources/33

stage="1-5"
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

# for rnnlm
lm_epochs=30
lm_batch_size=32
lm_num_workers=4

# for ngram lm
ngram=5

# decoding
eos_threshold=1
beam_size=16
nbest=8
ngram=5
lm_weight=0.2
test_sets="dev test"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  for name in data_aishell resource_aishell; do
    local/download_and_untar.sh $data $data_url $name
  done
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript data/$dataset
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training AM ..."
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

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: decoding ..."
  # decoding
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --gpu $gpu \
      --beam-size $beam_size \
      --nbest $nbest \
      --max-len 50 \
      --len-norm true \
      --dict exp/$dataset/$am_exp/dict \
      --log-suffix $name \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$name &
  done
  wait
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: training ngram LM ..."
  exp_dir=exp/aishell_v1/ngram && mkdir -p $exp_dir
  cat data/aishell_v1/train/text | awk '{$1=""; print}' > $exp_dir/train.text
  lmplz -o $ngram --text $exp_dir/train.text --arpa $exp_dir/${ngram}gram.arpa
  build_binary $exp_dir/${ngram}gram.arpa $exp_dir/${ngram}gram.arpa.bin
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: decoding (ngram) ..."
  for name in $test_sets; do
    dec_dir=${name}_${ngram}gram_$lm_weight
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --lm exp/aishell_v1/nnlm/$lm_exp \
      --gpu $gpu \
      --dict exp/$dataset/$am_exp/dict \
      --nbest $nbest \
      --lm exp/aishell_v1/ngram/${ngram}gram.arpa.bin \
      --lm-weight $lm_weight \
      --eos-threshold $eos_threshold \
      --max-len 50 \
      --len-norm true \
      --beam-size $beam_size \
      --lm-weight $lm_weight \
      --log-suffix $name \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$dec_dir &
  done
  wait
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: training RNNLM ..."
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

if [ $end -ge 7 ] && [ $beg -le 7 ]; then
  echo "Stage 7: decoding (RNNLM) ..."
  for name in $test_sets; do
    dec_dir=${name}_lm${lm_exp}_$lm_weight
    ./scripts/decode.sh \
      --score true \
      --text data/$dataset/$name/text \
      --lm exp/aishell_v1/nnlm/$lm_exp \
      --gpu $gpu \
      --dict exp/$dataset/$am_exp/dict \
      --nbest $nbest \
      --max-len 50 \
      --beam-size $beam_size \
      --lm-weight $lm_weight \
      --log-suffix $name \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$dec_dir &
  done
  wait
fi
