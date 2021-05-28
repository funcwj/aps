#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

dataset="chime4_unsuper"

. ./utils/parse_options.sh || exit 1;

[ $# -ne 2 ] && echo "Script format error: $0 <chime4-src-data> <cache-dir>" && exit 1

track_6ch=$1/data/audio/16kHz/isolated_6ch_track
cache_dir=$2

[ ! -d $track_6ch ] && echo "$0: Missing directory $track_6ch" && exit 1

mkdir -p $cache_dir/{tr05,dt05,et05}
for name in tr05 dt05 et05; do
  echo "$0: Merging wave to $cache_dir/$name ..."
  # Although our dataloader supports pipe style input, here we merge the single-channel audio to
  # multi-channel using sox
  find $track_6ch/${name}_*_{simu,real} -name "*.wav" | grep CH1 | \
    awk -F '[/.]' '{printf("%s %s\n", $(NF-2), $0)}' | sed 's:CH1:CH{1,3,4,5,6}:' | \
    awk -v dir=$cache_dir/$name '{printf("sox -M %s %s/%s.wav\n", $2, dir, $1)}' | bash
done

mkdir -p data/$dataset

find $(cd $cache_dir/tr05 && pwd) -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | \
  sed 's:.wav::' > data/$dataset/trn.scp
find $(cd $cache_dir/dt05 && pwd) -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | \
  sed 's:.wav::' > data/$dataset/dev.scp
find $(cd $cache_dir/et05 && pwd) -name "*.wav" | awk -F '/' '{printf("%s %s\n", $NF, $0)}' | \
  sed 's:.wav::' > data/$dataset/tst.scp

echo "$0: Prepare data done"
