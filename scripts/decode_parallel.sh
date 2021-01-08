#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

nj=20
cmd="utils/run.pl"
dict=""
space=""
nbest=1
channel=-1
max_len=100
penalty=0
beam_size=16
function="beam_search"
temperature=1
wav_norm=true
len_norm=true
am_tag="best"
lm_tag="best"
lm=""
lm_weight=0
spm=""

echo "$0 $*"

. ./utils/parse_options.sh || exit 1

[ $# -ne 4 ] && echo "Script format error: $0 <mdl-name> <exp-id> <tst-scp> <dec-dir>" && exit 1

mdl_id=$1
exp_id=$2

tst_scp=$3
dec_dir=$4

exp_dir=exp/$mdl_id/$exp_id
log_dir=$dec_dir/log && mkdir -p $log_dir

[ ! -f $tst_scp ] && echo "$0: missing test wave script: $tst_scp" && exit 0
[ ! -d $exp_dir ] && echo "$0: missing experiment directory: $exp_dir" && exit 0

wav_sp_scp=""
for n in $(seq $nj); do wav_sp_scp="$wav_sp_scp $log_dir/wav.$n.scp"; done

./utils/split_scp.pl $tst_scp $wav_sp_scp || exit 1

$cmd JOB=1:$nj $log_dir/decode.JOB.log \
  cmd/decode.py \
  $log_dir/wav.JOB.scp \
  $log_dir/beam${beam_size}.JOB.decode \
  --beam-size $beam_size \
  --am $exp_dir \
  --device-id -1 \
  --channel $channel \
  --am-tag $tag \
  --lm-tag $lm_tag \
  --dict "$dict" \
  --lm "$lm" \
  --spm "$spm" \
  --lm-weight $lm_weight \
  --penalty $penalty \
  --temperature $temperature \
  --space "$space" \
  --nbest $nbest \
  --dump-nbest $log_dir/beam${beam_size}.JOB.${nbest}best \
  --max-len $max_len \
  --function $function \
  --wav-norm $wav_norm \
  --len-norm $len_norm

cat $log_dir/beam${beam_size}.*.decode | \
  sort -k1 > $dec_dir/beam${beam_size}.decode
cat $log_dir/beam${beam_size}.*.${nbest}best | \
  sort -k1 > $dec_dir/beam${beam_size}.${nbest}best

echo "$0 $*: Done"
