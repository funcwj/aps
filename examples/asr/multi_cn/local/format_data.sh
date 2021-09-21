#!/usr/bin/env bash

set -eu

nj=4

[ $# -ne 1 ] && echo "Script format error: $0 <data-dir>" && exit 1

data_dir=$1

mkdir -p $data_dir/{train,dev,test}
# train
cat $data_dir/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train/wav.scp \
	> $data_dir/train/wav.scp
cat $data_dir/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train/text | \
	utils/tokenizer.py --unit char - $data_dir/train/text
utils/wav_duration.py --num-jobs $nj --output "time" $data_dir/train/wav.scp \
	$data_dir/train/utt2dur
# dev
cat $data_dir/{aidatatang,aishell,magicdata,thchs}/dev/wav.scp > $data_dir/dev/wav.scp
cat $data_dir/{aidatatang,aishell,magicdata,thchs}/dev/text | \
	utils/tokenizer.py --unit char - $data_dir/dev/text
utils/wav_duration.py --num-jobs $nj --output "time" $data_dir/dev/wav.scp \
	$data_dir/dev/utt2dur
# test
for x in aidatatang aishell magicdata thchs; do
	mkdir -p $data_dir/test/$x && cp $data_dir/$x/test/wav.scp $data_dir/test/$x
	utils/tokenizer.py --unit char $data_dir/$x/test/wav.scp $data_dir/test/$x/text
done
