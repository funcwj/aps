#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu
nj=16
stage="2-4"

# data
database_dir=/scratch/jwu/gigaspeech
archive_dir=/scratch/jwu/gigaspeech_ark
dataset=gigaspeech
train_subset=xl

# word piece
wp_name="wpm_5k"
wp_mode="unigram"
vocab_size=5000

# am
am_exp=1a
am_batch_size=96
am_epochs=120

# decoding
beam_size=8
nbest=$beam_size
len_norm=false
ctc_weight=0.2
test_sets="dev test"

. ./utils/parse_options.sh || exit 1

data_dir=data/$dataset
exp_dir=exp/$dataset/$am_exp

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

declare -A num_arks
num_arks=(
  ["train_xl"]=384
  ["train_l"]=128
  ["train_m"]=64
  ["train_s"]=32
  ["train_xs"]=16
  ["dev"]=4
)
train=train_$train_subset

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing the data ..."
  local/prep_data.sh --train-subset $train \
    --stage 1 --prefix "" $database_dir $data_dir
  # archive to wav.xx.ark for training
  # may take some time here
  for x in $train dev; do
    dst_dir=$archive_dir/$x && mkdir -p $dst_dir
    src_dir=$data_dir/$x
    cmd/archive_wav.py \
      --num-arks ${num_arks[$x]} \
      --segment $src_dir/segments \
      --num-jobs $nj \
      $src_dir/wav.scp \
      $dst_dir/wav.scp \
      $dst_dir/wav.ark && \
      cp $dst_dir/wav.scp $src_dir/wav_ark.scp
  done
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: tokenizing ..."
  # training
  ./utils/subword.sh --op "train" --mode $wp_mode --vocab-size $vocab_size \
    $data_dir/$train/text exp/$dataset/$wp_name
  cp exp/$dataset/$wp_name/dict $data_dir
  # wp encoding
  for x in dev $train; do
    ./utils/subword.sh --op "encode" --encode "piece" \
      $data_dir/$x/text exp/$dataset/$wp_name \
      > $data_dir/$x/token
  done
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training AM ..."
  # training am
  ./scripts/distributed_train.sh \
    --seed 888 \
    --gpu "0,1,2,3,4,5,6,7" \
    --distributed "torch" \
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers 32 \
    --prog-interval 250 \
    --eval-interval 2500 \
    am $dataset $am_exp
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: decoding ..."
  for name in $test_sets; do
    ./scripts/decode.sh \
      --score true \
      --text $data_dir/$name/text \
      --beam-size $beam_size \
      --max-len 150 \
      --segment $data_dir/$name/segments \
      --ctc-weight $ctc_weight \
      --len-norm $len_norm \
      --dict $exp_dir/dict \
      --nbest $nbest \
      --spm exp/$dataset/$wp_name/$wp_mode.model \
      $exp_dir $data_dir/$name/wav.scp \
      $exp_dir/$name &
  done
  wait
fi
