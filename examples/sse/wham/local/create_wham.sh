#!/usr/bin/env bash

set -eu

[ $# -ne 2 ] && echo "Script format error: $0 <wsj0-dir> <wham-out>" && exit 1

wsj0_dir=$(cd $1 && pwd)
wham_out=$(cd $2 && pwd)

cur_dir=$PWD

if [ ! -d $wham_out/wham_noise ]; then
  echo "Downloading wham_noise.zip to $wham_out ..."
  wget https://storage.googleapis.com/whisper-public/wham_noise.zip -P $wham_out
  unzip $wham_out/wham_noise.zip -d $wham_out && rm $wham_out/wham_noise.zip
fi

echo "Downloading wham_scripts.tar.gz to $wham_out ..."
wget https://storage.googleapis.com/whisper-public/wham_scripts.tar.gz -P $wham_out
tar -xzf $wham_out/wham_scripts.tar.gz -C $wham_out && rm $wham_out/wham_scripts.tar.gz

cd $wham_out/wham_scripts
# sed "s:'max', ::" create_wham_from_scratch.py | sed "s:, '8k'::" > create_wham_16k_min.py
python create_wham_from_scratch.py \
	--wsj0-root $wsj0_dir \
  --wham-noise-root $wham_out/wham_noise
	--output-dir $wham_out/wham_mix

cd $cur_dir && echo "$0: done"
