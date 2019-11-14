#!/usr/bin/env bash

# wujian@2019

set -eu

data_set=aishell_v1
data_dir=data/$data_set
beam_size=24

echo "$0 $@"

[ $# -ne 1 ] && echo "Script format error: $0 <exp-id>" && exit 1

exp_id=$1
exp_dir=exp/$data_set/$exp_id

cmd="/home/work_nfs/common/tools/pyqueue_tts.pl"
python=$(which python)

$cmd --gpu 1 decode.$exp_id.log \
  $python asr/decode.py \
    $data_dir/tst/wav.scp \
    $exp_dir/beam${beam_size}_decode.token \
    --beam-size $beam_size \
    --checkpoint $exp_dir \
    --device-id 0 \
    --nnet "las" \
    --max-len 50

cat $exp_set/beam${beam_size}_decode.token | \
  local/apply_map.pl -f 2- <(awk '{print $2"\t"$1}' $data_dir/dict) \
  > $exp_dir/beam${beam_size}_decode.trans
