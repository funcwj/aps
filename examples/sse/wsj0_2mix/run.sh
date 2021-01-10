#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage="1-3"
dataset="wsj0_2mix"
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

. ./utils/parse_options.sh || exit 1;

[ $# -ne 1 ] && echo "Script format error: $0 <wsj0-2mix-dir>" && exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

data_dir=$1

prepare_scp () {
  find $2 -name "*.$1" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | sed "s:.$1::"
}

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: preparing data ..."
  for x in "tr" "tt" "cv"; do [ ! -d $data_dir/$x ] && echo "$data_dir/$x not exists, exit ..." && exit 1; done
  data_dir=$(cd $data_dir && pwd)
  mkdir -p data/$dataset/{tr,cv,tt}
  for dir in tr cv tt; do
    # make mix.scp
    prepare_scp wav $data_dir/$dir/mix > data/$dataset/$dir/mix.scp
    # make spk{1,2}.scp
    prepare_scp wav $data_dir/$dir/s1 > data/$dataset/$dir/spk1.scp
    prepare_scp wav $data_dir/$dir/s2 > data/$dataset/$dir/spk2.scp
  done
  echo "$0: Prepare data done under data/$dataset"
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 2: training SS model ..."
  ./scripts/train.sh \
    --gpu $gpu --seed $seed \
    --epochs $epochs --batch-size $batch_size \
    --num-workers $num_workers \
    --eval-interval $eval_interval \
    --save-interval $save_interval \
    --prog-interval $prog_interval \
    --tensorboard $tensorboard \
    ss $dataset $exp
  echo "$0: Train model done under exp/$dataset/$exp"
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: evaluating ..."
  # generate separation audio under exp/$dataset/$exp/bss
  ./cmd/separate.py \
    --checkpoint exp/$dataset/$exp \
    --sr 8000 \
    --device-id $gpu \
    data/$dataset/tt/mix.scp \
    exp/$dataset/$exp/bss
  # remix
  mkdir -p exp/$dataset/$exp/bss/spk{1,2}
  prepare_scp wav exp/$dataset/$exp/bss | awk -v \
    dir=exp/$dataset/$exp/bss/spk1 '{printf("sox %s %s/%s.wav remix 1\n", $2, dir, $1)}' | bash
  prepare_scp wav exp/$dataset/$exp/bss | awk -v \
    dir=exp/$dataset/$exp/bss/spk2 '{printf("sox %s %s/%s.wav remix 2\n", $2, dir, $1)}' | bash
  # compute si-snr
  prepare_scp wav exp/$dataset/$exp/bss/spk1 > exp/$dataset/$exp/bss/spk1.scp
  prepare_scp wav exp/$dataset/$exp/bss/spk2 > exp/$dataset/$exp/bss/spk2.scp
  ./cmd/compute_ss_metric.py --sr 8000 --metric sisnr \
    data/$dataset/tt/spk1.scp,data/$dataset/tt/spk2.scp \
    exp/$dataset/$exp/bss/spk1.scp,exp/$dataset/$exp/bss/spk2.scp
fi
