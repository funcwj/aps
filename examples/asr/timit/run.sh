#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

timit_data=/scratch/jwu/TIMIT-LDC93S1/TIMIT
dataset="timit"
stage="1-3"
# training
gpu=0
exp=1a # load training configuration in conf/timit/1a.yaml
seed=777
epochs=100
tensorboard=false
batch_size=32
num_workers=4
prog_interval=100

# decoding
beam_size=8
len_norm=false
ctc_weight=0.2
nbest=$beam_size
test_sets="dev test"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

exp_dir=exp/$dataset/$exp
data_dir=data/$dataset

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  ./local/timit_data_prep.sh --dataset $dataset $timit_data
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 1: training AM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $epochs \
    --batch-size $batch_size \
    --tensorboard $tensorboard \
    --num-workers $num_workers \
    --prog-interval $prog_interval \
    --eval-interval -1 \
    am $dataset $exp
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: decoding ..."
  # decoding
  for name in $test_sets; do
    ./scripts/decode.sh \
      --gpu $gpu \
      --dict $exp_dir/dict \
      --nbest $nbest \
      --max-len 75 \
      --len-norm $len_norm \
      --ctc-weight $ctc_weight \
      --beam-size $beam_size \
      --dec-prefix beam${beam_size}_raw \
      $exp_dir $data_dir/$name/wav.scp \
      $exp_dir/dec_$name &
  done
  wait
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: compute wer ..."
  for name in $test_sets; do
    txt=beam${beam_size}_raw.decode
    hyp=$exp_dir/dec_$name/$txt
    # mapping
    local/timit_norm_trans.pl -i $hyp -m conf/$dataset/phones.60-48-39.map \
      -from 48 -to 39 > ${hyp}.39phn
    # wer
    ./cmd/compute_wer.py ${hyp}.39phn $data_dir/$name/text.39phn
  done
fi
