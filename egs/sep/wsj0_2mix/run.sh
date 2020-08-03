#!/usr/bin/env bash

# wujian@2019

set -eu

stage=1
dataset="wsj0_2mix"
exp_id="1a"
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

data_dir=$1

prepare_scp () {
  find $2 -name "*.$1" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | sed "s:.$1::"
}

if [ $stage -le 1 ]; then
    for x in "tr" "tt" "cv"; do [ ! -d $data_dir/$x ] && echo "$data_dir/$x not exists, exit ..." && exit 1; done
    data_dir=$(cd $data_dir && pwd)
    mkdir -p data/$dataset/{train,dev,tst}
    # make mix.scp
    prepare_scp wav $data_dir/tr/mix > data/$dataset/train/mix.scp
    prepare_scp wav $data_dir/cv/mix > data/$dataset/dev/mix.scp
    prepare_scp wav $data_dir/tt/mix > data/$dataset/test/mix.scp
    # make spk{1,2}.scp
    for spk in {1..2}; do
        prepare_scp wav $data_dir/tr/s$spk > data/$dataset/train/spk$spk.scp
        prepare_scp wav $data_dir/cv/s$spk > data/$dataset/dev/spk$spk.scp
        prepare_scp wav $data_dir/tt/s$spk > data/$dataset/tst/spk$spk.scp
    done
    echo "$0: Prepare data done under data/$dataset"
fi

if [ $stage -le 2 ]; then
    ./scripts/train_ss.sh \
        --gpu $gpu --seed $seed \
        --epochs $epochs --batch-size $batch_size \
        --num-workers $num_workers \
        --eval-interval $eval_interval \
        --save-interval $save_interval \
        --prog-interval $prog_interval \
        --tensorboard $tensorboard \
        $dataset $exp_id
    echo "$0: Train model done under exp/$dataset/$exp_id"
fi

if [ $stage -le 3 ]; then
    # generate separation audio under exp/$dataset/$exp_id/bss
    ./bin/eval_bss \
        --checkpoint exp/$dataset/$exp_id \
        --sr 8000 \
        --device-id $gpu \
        data/$dataset/tst/mix.scp \
        exp/$dataset/$exp_id/bss
    # remix
    mkdir -p exp/$dataset/$exp_id/bss/spk{1,2}
    prepare_scp wav exp/$dataset/$exp_id/bss | awk -v \
        dir=exp/$dataset/$exp_id/bss/spk1 '{printf("sox %s %s/%s.wav remix 1\n", $2, dir, $1)}' | bash
    prepare_scp wav exp/$dataset/$exp_id/bss | awk -v \
        dir=exp/$dataset/$exp_id/bss/spk2 '{printf("sox %s %s/%s.wav remix 2\n", $2, dir, $1)}' | bash
    # compute si-snr
    prepare_scp wav exp/$dataset/$exp_id/bss/spk1 > exp/$dataset/$exp_id/bss/spk1.scp
    prepare_scp wav exp/$dataset/$exp_id/bss/spk2 > exp/$dataset/$exp_id/bss/spk2.scp
    ./bin/compute_sisnr.py data/$dataset/tst/spk1.scp,data/$dataset/tst/spk2.scp \
        exp/$dataset/$exp_id/bss/spk1.scp,exp/$dataset/$exp_id/bss/spk2.scp
fi
