#!/usr/bin/env bash

# wujian@2020

set -eu

echo "$0: Formating chime4 data dir..."

nj=6
data_dir=data/chime4

for name in tr05 dt05; do
  mkdir -p $data_dir/$name
  cat $data_dir/${name}_{simu,real}_noisy/wav.scp $data_dir/${name}_orig_clean/wav.scp | \
    sort -k1 > $data_dir/$name/wav.scp
  cat $data_dir/${name}_{simu,real}_noisy/text $data_dir/${name}_orig_clean/text | \
    sort -k1 > $data_dir/$name/text
done

cp -rf $data_dir/tr05 $data_dir/train
cp -rf $data_dir/dt05 $data_dir/dev
rm -rf $data_dir/{tr05,dt05}

for name in train dev; do
  scripts/get_wav_dur.sh --nj $nj --output "time" \
    $data_dir/$name exp/utt2dur/$name
done

echo "$0: Format $data_dir done"
