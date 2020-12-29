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

# am
am_exp=1a
am_seed=888
am_batch_size=96
am_num_workers=8
am_epochs=70
am_prog_interval=100
am_eval_interval=3000

test_sets="test_dev93 test_eval92"

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

if [ $stage -le 3 ]; then
  # training am
  ./scripts/distributed_train.sh \
    --seed $am_seed \
    --gpu "0,1,2,3" \
    --num-process 4 \
    --distributed "torch" \
    --epochs $am_epochs \
    --num-workers $am_num_workers \
    --batch-size $am_batch_size \
    --prog-interval $am_prog_interval \
    --eval-interval $am_eval_interval \
    am wsj $am_exp
fi

if [ $stage -le 4 ]; then
  for name in $test_sets; do
    ./scripts/decode.sh \
      --log-suffix $name \
      --beam-size $beam_size \
      --max-len 150 \
      --dict data/wsj/dict \
      --nbest 8 \
      --spm exp/wsj/$wp_name/$wp_mode.model \
      wsj $am_exp \
      data/wsj/$name/wav.scp \
      exp/wsj/$am_exp/$name &
  done
  wait
  for name in $test_sets; do
    # WER
    ./cmd/compute_wer.py \
      exp/wsj/$am_exp/$name/beam$beam_size.decode \
      data/wsj/$name/text
  done
fi
