#!/bin/bash

# Copyright 2019 Xingyu Na
# Apache 2.0

set -eu

if [ $# != 2 ]; then
  echo "Usage: $0 <corpus-path> <data-path>"
  echo " $0 /export/a05/xna/data/stcmds data/stcmds"
  exit 1;
fi

corpus=$1/ST-CMDS-20170001_1-OS
data=$2

if [ ! -d $corpus ]; then
  echo "Error: $0 requires complete corpus"
  exit 1;
fi

echo "**** Creating ST-CMDS data folder ****"

mkdir -p $data/train

# find wav audio file for train

find $corpus -iname "*.wav" > $data/wav.list
n=`cat $data/wav.list | wc -l`
[ $n -ne 102600 ] && "echo Warning: expected 102600 data files, found $n"

cat $data/wav.list | awk -F'20170001' '{print $NF}' | awk -F'.' '{print $1}' > $data/utt.list
while read line; do
  tn=`dirname $line`/`basename $line .wav`.txt;
  cat $tn; echo;
done < $data/wav.list > $data/text.list

paste -d' ' $data/utt.list $data/wav.list | sort -k1 > $data/train/wav.scp
paste -d' ' $data/utt.list $data/text.list |\
  sed 's/，//g' |\
  tr '[a-z]' '[A-Z]' |\
  awk '{if (NF > 1) print $0;}' |\
  sort -k1 > $data/train/text

rm -r $data/{wav,utt,text}.list
