#!/bin/bash

# Copyright 2019 Xingyu Na
# Apache 2.0

set -eu

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/magicdata data/magicdata"
  exit 1;
fi

corpus=$1
data=$2

if [ ! -d $corpus/train ] || [ ! -d $corpus/dev ] || [ ! -d $corpus/test ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating magicdata data folder ****"

mkdir -p $data/{train,dev,test,tmp}

# find wav audio file for train, dev and test resp.
tmp_dir=$data/tmp
find $corpus -iname "*.wav" > $tmp_dir/wav.flist
n=`cat $tmp_dir/wav.flist | wc -l`
[ $n -ne 609552 ] && "echo Warning: expected 609552 data data files, found $n"

for x in train dev test; do
  grep -i "/$x/" $tmp_dir/wav.flist > $data/$x/wav.flist || exit 1;
  echo "Filtering data using found wav list and provided transcript for $x"
  awk -F '.wav' '{print $1}' local/magicdata_badlist | utils/filter_scp.pl --exclude -f 1 - \
    <(cat $data/$x/wav.flist|awk -F '/' '{print gensub(".wav", "", "g", $NF), $0}') | sort -k1 \
    > $data/$x/wav.scp
  sed '1d' $corpus/$x/TRANS.txt | awk -F '\t' '{print gensub(".wav","","g",$1), $3}' |\
    sed 's/！//g' | sed 's/？//g' |\
    sed 's/，//g' | sed 's/－//g' |\
    sed 's/：//g' | sed 's/；//g' |\
    sed 's/　//g' | sed 's/。//g' |\
    sed 's/`//g' | sed 's/,//g' |\
    sed 's/://g' | sed 's/?//g' |\
    sed 's/\///g' | sed 's/·//g' |\
    sed 's/\"//g' | sed 's/“//g' |\
    sed 's/”//g' | sed 's/\\//g' |\
    sed 's/…//g' | sed "s///g" |\
    sed 's/、//g' | sed "s///g" | sed 's/《//g' | sed 's/》//g' |\
    sed 's/\[//g' | sed 's/\]//g' | sed 's/FIL//g' | sed 's/SPK//' |\
    tr '[a-z]' '[A-Z]' |\
    awk '{if (NF > 1) print $0;}' | sort -k1 > $data/$x/text
done

rm -r $tmp_dir
