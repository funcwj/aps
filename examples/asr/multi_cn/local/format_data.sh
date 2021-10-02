#!/usr/bin/env bash

set -eu

nj=4

. utils/parse_options.sh || exit 1

[ $# -ne 1 ] && echo "Script format error: $0 <data-dir>" && exit 1

data_dir=$1

mkdir -p $data_dir/{train,dev,test}
# merge train
cat $data_dir/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train/wav.scp \
> $data_dir/train/wav.scp
cat $data_dir/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train/text | \
	utils/tokenizer.py --unit char --add-units "<sos>,<eos>,<unk>" \
	--dump-vocab $data_dir/dict - $data_dir/train/text
utils/wav_duration.py --num-jobs $nj --output "time" $data_dir/train/wav.scp \
  $data_dir/train/utt2dur
# merge dev
cat $data_dir/{aidatatang,aishell,magicdata,thchs}/dev/wav.scp > $data_dir/dev/wav.scp
cat $data_dir/{aidatatang,aishell,magicdata,thchs}/dev/text | \
	utils/tokenizer.py --unit char - $data_dir/dev/text
utils/wav_duration.py --num-jobs $nj --output "time" $data_dir/dev/wav.scp \
	$data_dir/dev/utt2dur
# dev & test
for x in aidatatang aishell magicdata thchs; do
	for name in dev test; do
	  mkdir -p $data_dir/$name/$x && cp $data_dir/$x/$name/wav.scp $data_dir/$name/$x
	  utils/tokenizer.py --unit char $data_dir/$x/$name/text $data_dir/$name/$x/text
	done
done
