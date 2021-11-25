#!/usr/bin/env bash

set -eu

[ $# -ne 2 ] && echo "Script format error: $0 <wham-data-dir> <script-dir>" && exit 1

# has {tr,tt,cv}
wham_dir=$1
data_dir=$2

gen_scp () {
  find $2 -name "*.$1" | awk -F '/' '{printf("%s\t%s\n", $NF, $0)}' | sed "s:.$1::"
}

for sr in 8k 16k; do
  for mode in max min; do
    mkdir -p $data_dir/wav${sr}_${mode}/{tr,cv,tt}
    for dir in tr cv tt; do
      for s in mix_{both,clean,single} s1 s2; do
        gen_scp wav $wham_dir/wav$sr/$mode/$dir/$s \
          > $data_dir/wav${sr}_${mode}/$dir/$s.scp
      done
    done
  done
done

echo "$0: Prepare data done under $data_dir"
