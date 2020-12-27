#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

wsj0="/home/jwu/doc/data/wsj0"
wsj1="/home/jwu/doc/data/wsj1"

stage=1

# word piece
wp_name="wpm_6k"
wp_mode="unigram"
vocab_size=6000

. ./utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
  ./local/wsj_data_prep.sh $wsj0/??-{?,??}.? $wsj1/??-{?,??}.?  || exit 1
  ./local/wsj_format_data.sh || exit 1
fi

if [ $stage -le 2 ]; then
  # training
  ./utils/subword.sh --op "train" --mode $wp_mode --vocab-size $vocab_size \
    data/wsj/train_si284/text exp/wsj/$wp_name
  cp exp/wsj/$wp_name/dict data/wsj
  # wp encoding
  for data in test_dev93 train_si284; do
    ./utils/subword.sh --op "encode" --encode "piece" \
      data/wsj/$data/text exp/wsj/$wp_name > data/wsj/$data/token
  done
fi
