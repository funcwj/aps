#!/usr/bin/env bash
# Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang).  Apache 2.0.
#           2016  LeSpeech (Author: Xingyu Na)

#This script pepares the data directory for thchs30 recipe.
#It reads the corpus and get wav.scp and transcriptions.

set -eu

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/data/thchs30 data/thchs30"
  exit 1;
fi

corpus_dir=$1
data=$2

echo "**** Creating THCHS-30 data folder ****"
mkdir -p $data/{train,dev,test}

# create wav.scp, text
for x in train dev test; do
  find $corpus_dir/$x -name "*.wav" | awk -F '/' '{print $NF}' |\
    sed 's:\.wav::' | sort -k1 > $data/$x/key
  awk -v dir=$corpus_dir/$x '{printf("TH%s %s/%s.wav\n", $1, dir, $1)}' \
    $data/$x/key > $data/$x/wav.scp
  echo "Generating text ..."
  while read uttid; do
    sed -n 1p $corpus_dir/data/$uttid.wav.trn;
  done < $data/$x/key > $data/$x/trans
  paste $data/$x/key $data/$x/trans |\
    awk '{printf("TH%s\n", $0)}' > $data/$x/text
  rm $data/$x/{key,trans}
done
