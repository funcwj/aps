#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# data
train_data=/home/jwu/doc/data/aishell_v2/AISHELL-2
valid_data=/home/jwu/doc/data/aishell_v2/AISHELL-2-Eval-Test
dataset="aishell_v2"

nj=4
stage=1
am_exp=1a

gpu="0,1,2,3"
seed=777
tensorboard=false
prog_interval=100
eval_interval=4000

# for am
am_epochs=100
am_batch_size=256
am_num_workers=32

# decoding
beam_size=16
len_norm=false
nbest=$beam_size
ctc_weight=0

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  local/prepare_data.sh $train_data/iOS/data data/$dataset/local/train data/$dataset/train
  scripts/get_wav_dur.sh --output "time" --nj $nj data/$dataset/train exp/$dataset/utt2dur
  for subset in DEV TEST; do
    for subtype in ANDROID IOS MIC; do
      name=$(echo ${subset}_${subtype} | tr '[:upper:]' '[:lower:]')
      local/prepare_data.sh $valid_data/$subset/$subtype data/$dataset/local/$name data/$dataset/$name
      scripts/get_wav_dur.sh --output "time" --nj $nj data/$dataset/$name exp/$dataset/utt2dur
    done
  done
  mkdir -p data/$dataset/dev
  for x in wav.scp utt2dur text; do cat data/$dataset/dev_*/$x | sort -k1 > data/$dataset/dev/$x; done
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training AM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers $am_num_workers \
    --tensorboard $tensorboard \
    --prog-interval $prog_interval \
    --eval-interval $eval_interval \
    --dev-batch-factor 4 \
    am $dataset $am_exp
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: decoding ..."
  # decoding
  for name in {dev,test}_{android,ios,mic}; do
    ./scripts/decode.sh \
      --gpu 0 \
      --text data/$dataset/$name/text \
      --score true \
      --beam-size $beam_size \
      --nbest $nbest \
      --max-len 50 \
      --ctc-weight $ctc_weight \
      --len-norm $len_norm \
      --dict exp/$dataset/$am_exp/dict \
      --log-suffix $name \
      $dataset $am_exp \
      data/$dataset/$name/wav.scp \
      exp/$dataset/$am_exp/$name &
  done
  wait
fi
