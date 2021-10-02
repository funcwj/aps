#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

nj=20
. ./utils/parse_options.sh || exit 1

[ $# -ne 1 ] && echo "Script format error: $0 <data-dir>" && exit 1

data_dir=$1
dataset=`basename $data_dir`

mkdir -p $data_dir/{train,dev}

for f in wav.scp utt2spk text spk2gender; do
  cat $data_dir/{train_clean_100,train_clean_360,train_other_500}/$f | sort -k1 > $data_dir/train/$f
  cat $data_dir/{dev_clean,dev_other}/$f | sort -k1 > $data_dir/dev/$f
done

for data in train dev; do
  echo "$0: prepare utt2dur for $data_dir/$data ..."
  utils/wav_duration.py --num-jobs $nj --output "time" $data_dir/$data/wav.scp $data_dir/$data/utt2dur
done

echo "$0: format data in $data_dir done"
