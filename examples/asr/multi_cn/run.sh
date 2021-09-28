#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu
stage="2-4"

database_dir=/scratch/jwu/openslr_cn
dataset=multi_cn
test_sets="aishell aidatatang magicdata thchs"

gpu="0,1,2,3"
seed=666
prog_interval=100
eval_interval=2500

# for am
am_exp=1a
am_epochs=100
am_batch_size=256
am_num_workers=32

# for lm
lm_exp=1a
lm_epochs=100
lm_batch_size=96
lm_num_workers=8

# decoding
beam_size=16
len_norm=false
nbest=$beam_size
lm_weight=0
ctc_weight=0.4

. ./utils/parse_options.sh || exit 1

data_dir=data/$dataset
beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: downloading & preparing the data ..."
  local/download_data.sh $database_dir
  local/prepare_data.sh $database_dir $data_dir
  local/format_data.sh $data_dir
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training AM ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $am_epochs \
    --batch-size $am_batch_size \
    --num-workers $am_num_workers \
    --tensorboard false \
    --prog-interval $prog_interval \
    --eval-interval $eval_interval \
    --dev-batch-factor 1 \
    am $dataset $am_exp
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  # decoding
  for data in dev test; do
    for name in $test_sets; do
      ./scripts/decode.sh \
        --gpu 0 \
        --text $data_dir/$data/$name/text \
        --score true \
        --beam-size $beam_size \
        --nbest $nbest \
        --max-len 80 \
        --ctc-weight $ctc_weight \
        --len-norm $len_norm \
        --dict exp/$dataset/$am_exp/dict \
        $dataset $am_exp \
        $data_dir/$data/$name/wav.scp \
        exp/$dataset/$am_exp/${data}_${name} &
    done
    wait
  done
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: training LM ..."
  ./scripts/train.sh \
    --gpu 0 \
    --seed $seed \
    --epochs $lm_epochs \
    --batch-size $lm_batch_size \
    --num-workers $lm_num_workers \
    --prog-interval $prog_interval \
    --eval-interval $eval_interval \
    lm $dataset $lm_exp
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: decoding (NNLM) ..."
  for data in dev test; do
    for name in $test_sets; do
      dec_dir=${data}_${name}_nnlm_$lm_weight
      ./scripts/decode.sh \
        --score true \
        --text data/$dataset/$data/$name/text \
        --gpu 0 \
        --dict exp/$dataset/$am_exp/dict \
        --nbest $nbest \
        --lm exp/$dataset/nnlm/$lm_exp \
        --lm-weight $lm_weight \
        --max-len 80 \
        --len-norm $len_norm \
        --beam-size $beam_size \
        --ctc-weight $ctc_weight \
        --lm-weight $lm_weight \
        $dataset $am_exp \
        $data_dir/$data/$name/wav.scp \
        exp/$dataset/$am_exp/$dec_dir &
    done
    wait
  done
fi
