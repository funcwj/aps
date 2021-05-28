#!/usr/bin/env bash

set -eu

[ $# -ne 2 ] && echo "Script format error: $0 <wham-data-dir> <script-dir>" && exit 1

# has {tr,tt,cv}
wham_dir=$1
data_dir=$2

prepare_scp () {
  find $2 -name "*.$1" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | sed "s:.$1::"
}

mkdir -p $data_dir/{tr,cv,tt}
for dir in tr cv tt; do
  prepare_scp wav $wham_dir/$dir/mix_both > $data_dir/$dir/mix.scp
  prepare_scp wav $wham_dir/$dir/s1 > $data_dir/$dir/spk1.scp
  prepare_scp wav $wham_dir/$dir/s2 > $data_dir/$dir/spk2.scp
  prepare_scp wav $wham_dir/$dir/noise > $data_dir/$dir/noise.scp
done

echo "$0: Prepare data done under $data_dir"
