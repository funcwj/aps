#!/usr/bin/env bash

# wujian@2019

set -eu

mode="word" # bpe, char, word, unigram
stage=1
encode="id"
decode="id"
vocab_size=6000

. ./local/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "$0: format error: <text> <exp-dir>" && exit 1

cmd="spm_train"
text=$1
exp_dir=$2

[ ! -f $text ] && echo "$0: missing $text" && exit 1

if ! command -v $cmd >/dev/null 2>&1; then
  echo "$0: compile https://github.com/google/sentencepiece please" && exit 1
fi

if [ $stage -eq 1 ]; then
  echo "$0: Training model" && mkdir -p $exp_dir
  cat $text | cut -d" " -f 2- > $exp_dir/trn.text && cd $exp_dir
  # keep sos/eos/unk same as asr
  $cmd --bos_id 1 \
    --bos_piece "<sos>" \
    --eos_id 2 \
    --eos_piece "<eos>" \
    --unk_id 0 \
    --unk_piece "<unk>" \
    --input="trn.text" \
    --vocab_size=$vocab_size \
    --model_type=$mode \
    --model_prefix=$mode
  awk '{printf("%s %d\n", $1, NR - 1)}' $mode.vocab > dict
fi

if [ $stage -eq 2 ]; then
  echo "$0: Encode input text using the model" >&2
  awk '{print $1}' $text > $exp_dir/enc.key 
  cat $text | cut -d" " -f 2- | spm_encode --model="$exp_dir/$mode.model" --output_format=$encode > $exp_dir/enc.val
  paste -d " " $exp_dir/enc.key $exp_dir/enc.val
fi


if [ $stage -eq 3 ]; then
  echo "$0: Decode input text using the model" >&2
  awk '{print $1}' $text > $exp_dir/dec.key 
  cat $text | cut -d" " -f 2- | spm_decode --model="$exp_dir/$mode.model" --input_format=$decode > $exp_dir/dec.val
  paste -d " " $exp_dir/dec.key $exp_dir/dec.val
fi

echo "$0 $@ done" >&2
