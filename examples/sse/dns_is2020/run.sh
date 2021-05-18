#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

# to get the data, run
# git clone https://github.com/microsoft/DNS-Challenge -b interspeech2020/master DNS-Challenge
dns_data=/home/jwu/doc/data/DNS-Challenge
dataset=dns_is2020
data_dir=data/$dataset

stage=2

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: simulate data ..."
  cd $dns_data && sed 's:\\:/:g' noisyspeech_synthesizer.cfg > noisyspeech_synthesizer.linux.cfg
  python noisyspeech_synthesizer_singleprocess.py --cfg noisyspeech_synthesizer.linux.cfg && cd -
fi

if [ $end -ge 2 ] && [ $beg -le 2 ]; then
  echo "Stage 1: prepare data ..."
  local/format_data.sh $dns_data $data_dir
fi
