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
max_len=500
min_len=1
max_len_ratio=1
min_len_ratio=0
len_norm=true
len_penalty=0
cov_penalty=0
cov_threshold=0
eos_threshold=0
beam_size=16
function="beam_search"
temperature=1
am_tag="best"
lm_tag="best"
lm=""
lm_weight=0
spm=""
dump_align=""
text=""
score=false

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

dec_prefix=beam${beam_size}_eos${eos_threshold}_lp${len_penalty}
[ $len_norm ] && dec_prefix=${dec_prefix}_norm

if [ -z $log_suffix ]; then
  log_suffix=$dec_prefix
else
  log_suffix=${dec_prefix}_${log_suffix}
fi

wav_sp_scp=""
for n in $(seq $nj); do wav_sp_scp="$wav_sp_scp $log_dir/wav.$n.scp"; done

./utils/split_scp.pl $tst_scp $wav_sp_scp || exit 1

$cmd JOB=1:$nj $log_dir/decode.$log_suffix.JOB.log \
  cmd/decode.py \
  $log_dir/wav.JOB.scp \
  $log_dir/${dec_prefix}.JOB.decode \
  --beam-size $beam_size \
  --am $exp_dir \
  --device-id -1 \
  --channel $channel \
  --am-tag $am_tag \
  --lm-tag $lm_tag \
  --dict "$dict" \
  --lm "$lm" \
  --spm "$spm" \
  --lm-weight $lm_weight \
  --temperature $temperature \
  --space "$space" \
  --nbest $nbest \
  --dump-nbest $log_dir/${dec_prefix}.JOB.${nbest}best \
  --dump-align "$dump_align" \
  --max-len $max_len \
  --min-len $min_len \
  --max-len-ratio $max_len_ratio \
  --min-len-ratio $min_len_ratio \
  --function $function \
  --len-norm $len_norm \
  --len-penalty $len_penalty \
  --cov-penalty $cov_penalty \
  --cov-threshold $cov_threshold \
  --eos-threshold $eos_threshold

cat $log_dir/${dec_prefix}.*.decode | sort -k1 > $dec_dir/${dec_prefix}.decode
cat $log_dir/${dec_prefix}.*.${nbest}best | sort -k1 > $dec_dir/${dec_prefix}.${nbest}best

if $score ; then
  [ -z $text ] && echo "for --score true, you must given --text <reference-transcription>" && exit -1
  ./cmd/compute_wer.py $dec_dir/${dec_prefix}.decode $text | \
    tee $dec_dir/${dec_prefix}.wer
fi

echo "$0 $*: Done"
