#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

wsj0_data=/scratch/jwu/wsj0
wham_data=/scratch/jwu/wham
dataset=wham

stage=2

exp="1a"
gpu=0
seed=777
epochs=100
tensorboard=false
batch_size=8
num_workers=4
eval_interval=-1
save_interval=-1
prog_interval=100

metric=sisnr

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

data_dir=data/$dataset

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: processing wsj0 ..."
  ./local/wv12wav.sh $wsj0_data
fi

# NOTE: we only create 16k min here
if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: simulating data ..."
  ./local/create_wham.sh $wsj0_data $wham_data
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: formating data ..."
  ./local/format_data.sh $wham_data/wham_mix/wav16k/min $data_dir
fi

cpt_dir=exp/$dataset/$exp
if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: training SS model ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --eval-interval $eval_interval \
    --save-interval $save_interval \
    --prog-interval $prog_interval \
    --tensorboard $tensorboard \
    ss $dataset $exp
  echo "$0: Train model done under $cpt_dir"
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: run speech separation ..."
  # generate separation audio under exp/$dataset/$exp/bss
  ./cmd/separate.py \
    --checkpoint $cpt_dir \
    --sr 16000 \
    --device-id $gpu \
    $data_dir/tt/mix.scp \
    $cpt_dir/bss
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: evaluate ..."
  # remix
  mkdir -p $cpt_dir/bss/spk{1,2}
  prepare_scp wav $cpt_dir/bss | awk -v dir=$cpt_dir/bss/spk1 \
    '{printf("sox %s %s/%s.wav remix 1\n", $2, dir, $1)}' | bash
  prepare_scp wav $cpt_dir/bss | awk -v dir=$cpt_dir/bss/spk2 \
    '{printf("sox %s %s/%s.wav remix 2\n", $2, dir, $1)}' | bash
  # compute si-snr
  prepare_scp wav $cpt_dir/bss/spk1 > $cpt_dir/bss/spk1.scp
  prepare_scp wav $cpt_dir/bss/spk2 > $cpt_dir/bss/spk2.scp
  ./cmd/compute_ss_metric.py --sr 16000 --metric $metric \
    $data_dir/tt/spk1.scp,$data_dir/tt/spk2.scp \
    $cpt_dir/bss/spk1.scp,$cpt_dir/bss/spk2.scp
fi
