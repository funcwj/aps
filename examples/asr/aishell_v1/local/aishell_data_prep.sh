#!/usr/bin/env bash

# Modified from https://github.com/kaldi-asr/kaldi/blob/master/egs/aishell/s5/local/aishell_data_prep.sh

set -eu

if [ $# != 3 ]; then
  echo "Usage: $0 <audio-path> <text-path> <dest-path>"
  exit 1
fi

aishell_audio_dir=$1
aishell_text_dir=$2
aishell_data_dir=$3
dataset=`basename $aishell_data_dir`

train_dir=$aishell_data_dir/local/train
dev_dir=$aishell_data_dir/local/dev
test_dir=$aishell_data_dir/local/test

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

# data directory check
if [ ! -d $aishell_audio_dir ] || [ ! -d $aishell_text_dir ]; then
  echo "Error: $0 requires two directory arguments"
  exit 1;
fi

# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/train" > $train_dir/wav.flist || exit 1;
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/dev" > $dev_dir/wav.flist || exit 1;
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/test" > $test_dir/wav.flist || exit 1;

n=`cat $train_dir/wav.flist $dev_dir/wav.flist $test_dir/wav.flist | wc -l`
[ $n -ne 141925 ] && echo Warning: expected 141925 data data files, found $n

# Transcriptions preparation
for dir in $train_dir $dev_dir $test_dir; do
  echo Preparing $dir transcriptions
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  utils/filter_scp.pl -f 1 $dir/utt.list $aishell_text_dir/*.txt > $dir/transcripts.txt
  awk '{print $1}' $dir/transcripts.txt | sort -u > $dir/utt.list
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  sort -u $dir/transcripts.txt | ./utils/tokenizer.pl - > $dir/text
done

mkdir -p $aishell_data_dir/{train,dev,test}
for f in wav.scp text; do
  cp $train_dir/$f $aishell_data_dir/train/$f || exit 1;
  cp $test_dir/$f $aishell_data_dir/test/$f || exit 1;
  cp $dev_dir/$f $aishell_data_dir/dev/$f || exit 1;
done

echo "$0: Prepare dictionary..."
./utils/tokenizer.py $aishell_data_dir/train/text /dev/null \
  --unit word --add-units "<sos>,<eos>,<unk>" --dump-vocab $aishell_data_dir/dict

echo "$0: Prepare utt2dur..."
for dir in train dev; do
  utils/wav_duration.py --num-jobs 10 --output "time" \
    $aishell_data_dir/$dir/wav.scp $aishell_data_dir/$dir/utt2dur
done

echo "$0: AISHELL data preparation succeeded"
exit 0
