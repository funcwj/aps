#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# to get the data, run
# git clone https://github.com/microsoft/DNS-Challenge -b interspeech2020/master DNS-Challenge
dns_data=/home/jwu/doc/data/DNS-Challenge
dataset=dns_is2020
data_dir=data/$dataset

stage=2

# training cfg
exp="1a"
gpu=0
seed=666
epochs=100
batch_size=32
num_workers=4
eval_interval=-1
save_interval=-1
prog_interval=100
metric="pesq stoi"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: simulate data ..."
  cd $dns_data && sed 's:\\:/:g' noisyspeech_synthesizer.cfg > noisyspeech_synthesizer.linux.cfg
  python noisyspeech_synthesizer_singleprocess.py --cfg noisyspeech_synthesizer.linux.cfg && cd -
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 1: prepare data ..."
  local/format_data.sh $dns_data $data_dir
fi

if [ $end -ge 3 ] && [ $beg -le 3 ]; then
  echo "Stage 3: training SE model ..."
  ./scripts/train.sh \
    --gpu $gpu \
    --seed $seed \
    --epochs $epochs \
    --batch-size $batch_size \
    --num-workers $num_workers \
    --eval-interval $eval_interval \
    --save-interval $save_interval \
    --prog-interval $prog_interval \
    ss $dataset $exp
fi

if [ $end -ge 4 ] && [ $beg -le 4 ]; then
  echo "Stage 4: run speech enhancement ..."
  for subdir in `ls $data_dir`; do
    ./cmd/separate.py \
      --checkpoint exp/$dataset/$exp \
      --sr 16000 \
      --device-id $gpu \
      $data_dir/$subdir/noisy.scp \
      exp/$dataset/$exp/$subdir
  done
fi

if [ $end -ge 5 ] && [ $beg -le 5 ]; then
  echo "Stage 5: evaluate ..."
  for subdir in `ls $data_dir`; do
    find exp/$dataset/$exp/$subdir -name "*.wav" | \
      awk -F '/' '{printf("%s %s\n", $NF, $0)}' | \
      sed 's:\.wav::' > exp/$dataset/$exp/$subdir/enhan.scp
  done
  for subdir in `ls $data_dir/test_synt_*`; do
    for cur_metric in $metric; do
      ./cmd/compute_ss_metric.py --sr 16000 \
        exp/$dataset/$exp/$subdir/enhan.scp \
        $data_dir/$subdir/clean.scp
    done
  done
fi
