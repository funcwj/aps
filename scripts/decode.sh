#!/usr/bin/env bash

# Copyright 2019 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

gpu=0
dict=""
space=""
nbest=1
channel=-1
max_len=100
beam_size=16
batch_size=""
function="beam_search"
penalty=0
wav_norm=true
len_norm=true
temperature=1
lm=""
lm_weight=0
spm=""
log_suffix=""

echo "$0 $*"

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
[ ! -z $log_suffix ] && log_suffix=${log_suffix}.

if [ -z $batch_size ]; then
  cmd/decode.py \
    $tst_scp \
    $dec_dir/beam${beam_size}.decode \
    --beam-size $beam_size \
    --checkpoint $exp_dir \
    --device-id $gpu \
    --channel $channel \
    --dict "$dict" \
    --lm "$lm" \
    --spm "$spm" \
    --penalty $penalty \
    --temperature $temperature \
    --lm-weight $lm_weight \
    --space "$space" \
    --nbest $nbest \
    --dump-nbest $dec_dir/beam${beam_size}.${nbest}best \
    --max-len $max_len \
    --wav-norm $wav_norm \
    --len-norm $len_norm \
    --function $function \
    > $mdl_id.decode.$exp_id.${log_suffix}log 2>&1
else
  cmd/decode_batch.py \
    $tst_scp \
    $dec_dir/beam${beam_size}.decode \
    --beam-size $beam_size \
    --batch-size $batch_size \
    --checkpoint $exp_dir \
    --device-id $gpu \
    --channel $channel \
    --dict "$dict" \
    --lm "$lm" \
    --spm "$spm" \
    --space "$space" \
    --penalty $penalty \
    --temperature $temperature \
    --lm-weight $lm_weight \
    --nbest $nbest \
    --dump-nbest $dec_dir/beam${beam_size}.${nbest}best \
    --max-len $max_len \
    --wav-norm $wav_norm \
    --len-norm $len_norm \
    > $mdl_id.decode.$exp_id.${log_suffix}log 2>&1
fi

echo "$0 $*: Done"
