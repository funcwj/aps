#!/usr/bin/env bash

# wujian@2019

set -eu

stage=1
max_len=200
beam_size=16
normalized=true

echo "$0 $@"

. ./local/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "Script format error: $0 <data-set> <exp-id>" && exit 1

data=$1
exp_id=$2

data_dir=data/$data
exp_dir=exp/$data/$exp_id

[ ! -d $data_dir/tst ] && echo "$0: missing test directory: $data_dir/tst" && exit 0

cmd="/home/work_nfs/common/tools/pyqueue_tts.pl"
python=$(which python)

if [ $stage -le 1 ]; then
  $cmd --gpu 1 $data.decode.$exp_id.log \
    $python asr/decode.py \
    $data_dir/tst/wav.scp \
    $exp_dir/beam${beam_size}_decode.token \
    --beam-size $beam_size \
    --checkpoint $exp_dir \
    --device-id 0 \
    --nnet "las" \
    --normalized $normalized \
    --vectorized false \
    --max-len $max_len
fi

if [ $stage -le 2 ]; then
  cat $exp_dir/beam${beam_size}_decode.token | \
    local/token2idx.pl <(awk '{print $2"\t"$1}' $data_dir/dict) \
    > $exp_dir/beam${beam_size}_decode.trans
fi