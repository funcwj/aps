#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

op="train"
mode="unigram" # bpe, char, word, unigram
encode="piece"
decode="piece"
vocab_size=6000

. utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "$0: format error: <text> <exp-dir>" && exit 1

cmd="spm_train"
text=$1
exp_dir=$2

[ ! -f $text ] && echo "$0: missing $text" && exit 1

if ! command -v $cmd >/dev/null 2>&1; then
  echo "$0: compile https://github.com/google/sentencepiece please" && exit 1
fi

case $op in
  "train" )
    echo "$0: Training model using $text ..." && mkdir -p $exp_dir
    cat $text | cut -d" " -f 2- > $exp_dir/trn.text && cd $exp_dir
    # keep sos/eos/unk same as asr
    $cmd --bos_id 1 --bos_piece "<sos>" --eos_id 2 --eos_piece "<eos>" \
      --unk_id 0 --unk_piece "<unk>" --unk_surface "<unk>" \
      --input="trn.text" --vocab_size=$vocab_size --model_type=$mode --model_prefix=$mode
    sort -k 1 $mode.vocab | awk '{printf("%s %d\n", $1, NR - 1)}' > dict && rm trn.text
    ;;
  "encode" )
    echo "$0: Encoding $text using the model $exp_dir/$mode.model ..." >&2
    awk '{print $1}' $text > $exp_dir/enc.key
    cat $text | awk '{$1=""; print $0;}' | spm_encode --model=$exp_dir/$mode.model \
      --output_format=$encode > $exp_dir/enc.val
    paste -d " " $exp_dir/enc.key $exp_dir/enc.val && rm $exp_dir/enc.{key,val}
    ;;
  "decode" )
    echo "$0: Decoding $text using the model $exp_dir/$mode.model ..." >&2
    awk '{print $1}' $text > $exp_dir/dec.key
    cat $text | awk '{$1=""; print $0;}' | spm_decode --model=$exp_dir/$mode.model \
      --input_format=$decode > $exp_dir/dec.val
    paste -d " " $exp_dir/dec.key $exp_dir/dec.val && rm $exp_dir/dec.{key,val}
    ;;
  * )
    echo "$0: Unknown parameter --op $op" && exit 1
    ;;
esac

echo "$0 $@ done" >&2
