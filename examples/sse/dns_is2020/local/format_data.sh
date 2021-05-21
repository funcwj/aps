#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

[ $# -ne 2 ] && echo "format error: $0 <dns-corpus> <data-dir>" && exit 0

corpus=$1
data_dir=$2

# four test sets
mkdir -p $data_dir/{test_blind,test_real,test_synt_with_reverb,test_synt_no_reverb}

find $corpus/datasets/blind_test_set -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' \
  | sed 's:\.wav::' > $data_dir/test_blind/noisy.scp
find $corpus/datasets/test_set/real_recordings -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' \
  | sed 's:\.wav::' > $data_dir/test_real/noisy.scp


prepare_clean() {
  wave_dir=$1
  dump_dir=$2
  find $wave_dir -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | sed 's:\.wav::' > $dump_dir/clean.scp
  awk '{print $1}' $dump_dir/clean.scp | awk -F '_' '{printf("%s_%s\n", $2, $3)}' > $dump_dir/clean.key
  awk '{print $2}' $dump_dir/clean.scp > $dump_dir/clean.val
  paste $dump_dir/clean.{key,val} | sort -k1 > $dump_dir/clean.scp && rm $dump_dir/clean.{key,val}
}

prepare_noisy() {
  wave_dir=$1
  dump_dir=$2
  find $wave_dir -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | sed 's:\.wav::' > $dump_dir/noisy.scp
  awk '{print $1}' $dump_dir/noisy.scp | awk -F '_' '{printf("%s_%s\n", $(NF-1), $NF)}' > $dump_dir/noisy.key
  awk '{print $2}' $dump_dir/noisy.scp > $dump_dir/noisy.val
  paste $dump_dir/noisy.{key,val} | sort -k1 > $dump_dir/noisy.scp && rm $dump_dir/noisy.{key,val}
}

echo "$0: done with $data_dir/{test_blind,test_real}"

for name in with_reverb no_reverb; do
  # process clean
  prepare_clean $corpus/datasets/test_set/synthetic/$name/clean $data_dir/test_synt_$name
  # process noisy
  prepare_noisy $corpus/datasets/test_set/synthetic/$name/noisy $data_dir/test_synt_$name
done

echo "$0: done with $data_dir/{test_synt_with_reverb,test_synt_no_reverb}"

# training

mkdir -p $data_dir/simu_all
prepare_clean $corpus/clean $data_dir/simu_all
prepare_noisy $corpus/noisy $data_dir/simu_all

total=$(cat $data_dir/simu_all/clean.scp | wc -l)
dev_utts=$(python -c "print(int(0.1*$total))")
echo "$0: $dev_utts utterances are used as dev sets"
awk '{print $1}' $data_dir/simu_all/clean.scp | shuf | head -n $dev_utts > $data_dir/simu_all/dev.key
mkdir -p $data_dir/{train,dev}

for name in clean noisy; do
  utils/filter_scp.pl -f 1 $data_dir/simu_all/dev.key $data_dir/simu_all/$name.scp > $data_dir/dev/$name.scp
  utils/filter_scp.pl -f 1 --exclude $data_dir/simu_all/dev.key $data_dir/simu_all/$name.scp > $data_dir/train/$name.scp
done

echo "$0: done with $data_dir/{train,dev}"
