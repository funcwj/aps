#!/usr/bin/env bash

set -eu

sr=16k
spk=2

. ./utils/parse_options.sh || exit 1

[ $# -ne 2 ] && echo "$0: script format error: <librimix_wav_dir> <local_data_dir>" && exit 1

librimix_wav_dir=$1
data_dir=$2

prep_scp () {
  find $2 -name "*.$1" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | sed "s:.$1::"
}

for mode in max min; do
  dir=$librimix_wav_dir/Libri${spk}Mix/wav$sr/$mode
  [ ! -d $dir ] && echo "$0: $dir does't exists, please run local/simu_librimix.sh first" && exit 1

  # e.g., 2spk_16k_max
  name=${spk}spk_${sr}_${mode} && mkdir -p $data_dir/$name
  for dset in dev test train-100 train-360; do
    mkdir -p $data_dir/$name/$dset
    for s in mix_{both,clean,single} noise s1 s2; do
      prep_scp wav $dir/$dset/$s > $data_dir/$name/$dset/$s.scp
    done
  done
  echo "$0: prepare $sr & $spk speaker data done in $data_dir/$name"
done
