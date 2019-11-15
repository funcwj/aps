#!/usr/bin/env bash

# wujian@2019

set -eu

beam_size=24

echo "$0 $@"

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2

data_dir=data/$data
exp_dir=exp/$data/$exp_id

cmd="/home/work_nfs/common/tools/pyqueue_tts.pl"
python=$(which python)

$cmd --gpu 1 $data_set.decode.$exp_id.log \
  $python asr/decode.py \
    $data_dir/tst/wav.scp \
    $exp_dir/beam${beam_size}_decode.token \
    --beam-size $beam_size \
    --checkpoint $exp_dir \
    --device-id 0 \
    --nnet "las" \
    --max-len 50

cat $exp_dir/beam${beam_size}_decode.token | \
  local/apply_map.pl -f 2- <(awk '{print $2"\t"$1}' $data_dir/dict) \
  > $exp_dir/beam${beam_size}_decode.trans
