#!/usr/bin/env bash

# wujian@2020

set -eu

stage=1
data=/scratch/jwu/LibriSpeech

# word piece
wp_name="wpm_6k"
wp_mode="unigram"
vocab_size=6000

# am
am_exp=1a
am_seed=888
am_batch_size=96
am_num_workers=16
am_epochs=70
am_prog_interval=100
am_eval_interval=3000

# lm
lm_exp=1a
lm_seed=777
lm_batch_size=96
lm_eval_interval=50000
lm_epochs=30
lm_num_workers=4

# decoding
beam_size=16
lm_weight=0.2

. ./utils/parse_options.sh || exit 1

# NOTE: download Librispeech data first
if [ $stage -le 1 ]; then
  for x in dev-{clean,other} test-{clean,other} train-clean-{100,360} train-other-500; do
    ./local/data_prep.sh $data/$x data/librispeech/$(echo $x | sed s/-/_/g)
  done
  ./local/format_data.sh --nj 10 data/librispeech
fi

if [ $stage -le 2 ]; then
  # training
  ./utils/subword.sh --op "train" --mode $wp_mode --vocab-size $vocab_size \
    data/librispeech/train/text exp/librispeech/$wp_name
  cp exp/librispeech/$wp_name/dict data/librispeech
  # wp encoding
  for data in dev train; do
    ./utils/subword.sh --op "encode" --encode "piece" \
      data/librispeech/$data/text exp/librispeech/$wp_name \
      > data/librispeech/$data/wp6k
  done
fi

if [ $stage -le 3 ]; then
  # training am
	./scripts/distributed_train_am.sh \
		--seed $am_seed \
		--gpu "0,1,2,3" \
    --num-process 4 \
    --distributed "torch" \
		--epochs $am_epochs \
		--num-workers $am_num_workers \
		--batch-size $am_batch_size \
		--prog-interval $am_prog_interval \
    --eval-interval $am_eval_interval \
		librispeech $am_exp
fi

if [ $stage -le 4 ]; then
  for data in test_clean test_other; do
    ./scripts/decode.sh \
      --log-suffix $data \
      --beam-size $beam_size \
      --max-len 150 \
      --dict data/librispeech/dict \
      --nbest 8 \
      librispeech $am_exp \
      data/librispeech/$data/wav.scp \
      exp/librispeech/$am_exp/$data &
  done
  wait
  for data in test_clean test_other; do
    # wp decoding
    ./utils/subword.sh \
      --op "decode" \
      --decode piece \
      exp/librispeech/$am_exp/$data/beam16.decode \
      exp/librispeech/$wp_name > exp/librispeech/$am_exp/$data/beam16.decode.final
    # WER
    ./bin/compute_wer.py \
      exp/librispeech/$am_exp/$data/beam16.decode.final \
      data/librispeech/$data/text
  done
fi

if [ $stage -le 5 ]; then
  # prepare lm data
  [ ! -f $data/lm/librispeech-lm-norm.txt ] && \
  mkdir $data/lm && cd $data/lm && wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz && \
    gunzip librispeech-lm-norm.txt.gz && cd -
  mkdir -p data/librispeech/lm
  cat data/librispeech/{train,dev}/wp6k | sort -k1 > data/librispeech/lm/dev.wp6k
  awk 'printf("utt-%d %s\n", NR, $0)' $data/lm/librispeech-lm-norm.txt > data/librispeech/lm/train.text
  ./utils/subword.sh --op "encode" --encode "piece" \
    data/librispeech/lm/train.text exp/librispeech/$wp_name \
    > data/librispeech/lm/train.wp6k
fi

if [ $stage -le 6 ]; then
  # training lm
	./scripts/train_lm.sh \
		--seed $lm_seed \
		--epochs $lm_epochs \
		--num-workers $lm_num_workers \
		--batch-size $lm_batch_size \
    --eval-interval $lm_eval_interval \
		librispeech $lm_exp
fi

if [ $stage -le 7 ]; then
  dec_name=${name}_lm${lm_exp}_$lm_weight
  # shallow fusion
  for name in test_clean test_other; do
    ./scripts/decode.sh \
      --log-suffix $name \
      --lm exp/librispeech/nnlm/$lm_exp \
      --lm-weight $lm_weight \
      --beam-size $beam_size \
      --max-len 150 \
      --dict data/librispeech/dict \
      --nbest 8 \
      librispeech $am_exp \
      data/librispeech/$name/wav.scp \
      exp/librispeech/$am_exp/$dec_name &
  done
  wait
  for name in test_clean test_other; do
    # wp decoding
    ./utils/subword.sh \
      --op "decode" \
      --decode piece \
      exp/librispeech/$am_exp/$dec_name/beam16.decode \
      exp/librispeech/$wp_name \
      > exp/librispeech/$am_exp/$dec_name/beam16.decode.final
    # WER
    ./bin/compute_wer.py \
      exp/librispeech/$am_exp/$dec_name/beam16.decode.final \
      data/librispeech/$name/text
  done
fi
