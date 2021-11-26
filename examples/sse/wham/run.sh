#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=2
wsj0_data=/scratch/jwu/wsj0
wham_data=/scratch/jwu/wham
dataset=wham
sr=16k # 8k,16k

# train
exp=1a_bss_c_16k_max
gpu=0
seed=777
epochs=100
tensorboard=false
batch_size=8
num_workers=4

# evaluate
metric=sisnr
eval_data=mix_clean
eval_mode=max # {min,max}

. ./utils/parse_options.sh || exit 1

# check sr
case $sr in
  "16k" )
  sr_num=16000
  ;;
  "8k" )
  sr_num=8000
  ;;
  * )
  echo "$0: Unknown sr: $sr" && exit 1
  ;;
esac

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
  mkdir -p $wham_data && ./local/create_wham.sh $wsj0_data $wham_data
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: formating data ..."
  ./local/format_data.sh $wham_data/wham_mix $data_dir
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
    --tensorboard $tensorboard \
    --eval-interval -1 \
    --save-interval -1 \
    --prog-interval 100 \
    ss $dataset $exp
  echo "$0: Train model done under $cpt_dir"
fi

# wav{8k,16k}_{min,max}
eval_dir=$data_dir/wav${sr}_${eval_mode}
if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: run BSS on ${eval_data}.scp ..."
  for data in cv tt; do
    ./cmd/separate.py \
      --checkpoint $cpt_dir \
      --sr $sr_num \
      --device-id $gpu \
      $eval_dir/$data/${eval_data}.scp \
      $cpt_dir/bss_$data
  done
fi

if [ $end -ge 6 ] && [ $beg -le 6 ]; then
  echo "Stage 6: evaluate ..."
  for data in cv tt; do
    for index in 1 2; do
      find $cpt_dir/bss_$data -name "*.wav" | \
        awk -v ch=$index -F '/' '{printf("%s sox %s -t wav - remix %d |\n", $NF, $0, ch)}' | \
        sed "s:.wav::" > $cpt_dir/bss_$data/s${index}.scp
    done
    ./cmd/compute_ss_metric.py \
      --sr $sr_num \
      --metric $metric \
      $eval_dir/$data/s1.scp,$eval_dir/$data/s2.scp \
      $cpt_dir/bss_$data/s1.scp,$cpt_dir/bss_$data/s2.scp
  done
fi
