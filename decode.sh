#!/usr/bin/env bash

# wujian@2019

set -eu

data_dir=data/aishell_v1
beam_size=24

echo "$0 $@"

[ $# -ne 1 ] && echo "Script format error: $0 <exp-id>" && exit 1

exp_id=$1

cmd="/home/work_nfs/common/tools/pyqueue_asr.pl"
python=$(which python)

$cmd --gpu 1 decode.$exp_id.log \
  $python asr/decode.py \
    $data_dir/tst/wav.scp \
    exp/aishell_v1/$exp_id/beam${beam_size}_decode.token \
    --beam-size $beam_size \
    --checkpoint exp/aishell_v1/$exp_id \
    --device-id 0 \
    --max-len 100

cat exp/aishell_v1/$exp_id/beam${beam_size}_decode.token | \
  local/apply_map.pl -f 2- <(awk '{print $2"\t"$1}' $data_dir/dict) \
  > exp/aishell_v1/$exp_id/beam${beam_size}_decode.trans
