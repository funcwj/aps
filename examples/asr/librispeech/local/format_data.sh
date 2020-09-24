#!/usr/bin/env bash

# wujian@2020

set -eu

nj=20
. ./utils/parse_options.sh || exit 1

[ $# -ne 1 ] && echo "Script format error: $0 <data-dir>" && exit 1

data_dir=$1

mkdir -p $data_dir/{train,dev}

for f in wav.scp utt2spk spk2utt text spk2gender; do
  cat $data_dir/{train_clean_100,train_clean_360,train_other_500}/$f | sort -k1 > $data_dir/train/$f
  cat $data_dir/{dev_clean,dev_other}/$f | sort -k1 > $data_dir/dev/$f
done

for data in train dev; do
  scripts/get_wav_dur.sh --nj $nj --output "time" $data_dir/$data exp/utt2dur/$data
done

echo "$0: format data in $data_dir done"
