#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=1
dataset=librimix
librimix_data_dir=/mnt/jwu/librimix

# data setup
sr=16k
spk=2

# train
gpu="0,1,2,3"
exp=1a_2spk_c_16k_min
epochs=100
batch_size=32
tensorboard=false

# evaluate
metric=sisnr
eval_data=mix_both
# train on min and eval on max
eval_mode=max

. ./utils/parse_options.sh || exit 1

# e.g., 2spk_16k_max
data_dir=data/$dataset
eval_dir=$data_dir/${spk}spk_${sr}_${eval_mode}
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

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: simulate LibriMix data ..."
  local/simu_librimix.sh $librimix_data_dir
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: format LibriMix data ..."
  local/prep_data.sh --spk $spk --sr $sr \
    $librimix_data_dir $data_dir
fi

cpt_dir=exp/$dataset/$exp
if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training BSS model ..."
  ./scripts/distributed_train.sh \
    --gpu $gpu \
    --seed 666 \
    --epochs $epochs \
    --batch-size $batch_size \
    --tensorboard $tensorboard \
    --num-workers 16 \
    --prog-interval 250 \
    ss $dataset $exp
  echo "$0: Train model done under $cpt_dir"
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: run BSS ..."
  for name in dev test; do
    ./cmd/separate.py \
      --checkpoint $cpt_dir \
      --sr $sr_num \
      --device-id 0 \
      $eval_dir/$name/${eval_data}.scp \
      $cpt_dir/bss_$name
  done
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: evaluate ..."
  spk_index=$(seq $spk)
  for name in dev test; do
    for index in $spk_index; do
      find $cpt_dir/bss_$name -name "*.wav" | \
        awk -v ch=$index -F '/' '{printf("%s sox %s -t wav - remix %d |\n", $NF, $0, ch)}' | \
        sed "s:.wav::" > $cpt_dir/bss_$name/s${index}.scp
    done
    est_list=$eval_dir/$name/s1.scp,$eval_dir/$name/s2.scp
    ref_list=$cpt_dir/bss_$name/s1.scp,$cpt_dir/bss_$name/s2.scp
    if [ $spk -eq 3]; then
      est_list="$est_list,$eval_dir/$name/s3.scp"
      ref_list="$ref_list,$cpt_dir/bss_$name/s3.scp"
    fi
    ./cmd/compute_ss_metric.py \
      --sr $sr_num \
      --metric $metric \
      $est_list $ref_list
  done
fi
