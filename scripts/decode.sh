#!/usr/bin/env bash

# wujian@2019

set -eu

gpu=0
dict=""
space=""
nbest=1
channel=-1
max_len=100
beam_size=16
batch_size=1
normalized=true
lm=""
lm_weight=0

echo "$0 $@"

. ./utils/parse_options.sh || exit 1

[ $# -ne 4 ] && echo "Script format error: $0 <mdl-name> <exp-id> <tst-scp> <dec-dir>" && exit 1

mdl_id=$1
exp_id=$2

tst_scp=$3
dec_dir=$4

exp_dir=exp/$mdl_id/$exp_id

[ ! -f $tst_scp ] && echo "$0: missing test wave script: $tst_scp" && exit 0
[ ! -d $exp_dir ] && echo "$0: missing experiment directory: $exp_dir" && exit 0

mkdir -p $dec_dir
if [ $batch_size -eq 1 ]; then
  bin/decode.py \
    $tst_scp \
    $dec_dir/beam${beam_size}.decode \
    --beam-size $beam_size \
    --checkpoint $exp_dir \
    --device-id $gpu \
    --channel $channel \
    --dict "$dict" \
    --lm "$lm" \
    --lm-weight $lm_weight \
    --space "$space" \
    --nbest $nbest \
    --dump-nbest $dec_dir/beam${beam_size}.${nbest}best \
    --max-len $max_len \
    --normalized $normalized \
    --vectorized true \
    > $mdl_id.decode.$exp_id.log 2>&1
else
  bin/decode_batch.py \
    $tst_scp \
    $dec_dir/beam${beam_size}.decode \
    --beam-size $beam_size \
    --batch-size $batch_size \
    --checkpoint $exp_dir \
    --device-id $gpu \
    --channel $channel \
    --dict "$dict" \
    --space "$space" \
    --nbest $nbest \
    --dump-nbest $dec_dir/beam${beam_size}.${nbest}best \
    --max-len $max_len \
    --normalized $normalized \
    > $mdl_id.decode.$exp_id.log 2>&1
fi

echo "$0 $@: Done"
