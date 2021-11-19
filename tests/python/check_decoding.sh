#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

# checkpoints may out of date

set -eu

# test decoding for att
# diable on github workflow
cpt_dir=data/checkpoint/aishell_att_1a
../cmd/decode.py $cpt_dir/egs.scp - \
    --beam-size 16 \
    --nbest 8 \
    --dump-nbest beam.nbest \
    --am $cpt_dir \
    --device-id -1 \
    --channel -1 \
    --dict $cpt_dir/dict \
    --max-len 50 \
    --len-norm true
../cmd/decode.py $cpt_dir/egs.scp - \
    --dump-nbest greedy.nbest \
    --am $cpt_dir \
    --device-id -1 \
    --channel -1 \
    --dict $cpt_dir/dict \
    --function "greedy_search" \
    --max-len 50 \
    --len-norm true
../cmd/decode_batch.py $cpt_dir/egs.scp - \
    --dump-nbest batch.nbest \
    --beam-size 16 \
    --nbest 8 \
    --batch-size 1 \
    --am $cpt_dir \
    --device-id -1 \
    --channel -1 \
    --dict $cpt_dir/dict \
    --max-len 50 \
    --len-norm true

# test decoding for rnnt
cpt_dir=data/checkpoint/timit_rnnt_1a
../cmd/decode.py $cpt_dir/egs.scp - \
    --beam-size 4 \
    --am $cpt_dir \
    --device-id -1 \
    --channel -1 \
    --dict $cpt_dir/dict \
    --max-len 50 \
    --len-norm true
../cmd/decode.py $cpt_dir/egs.scp - \
    --function "greedy_search" \
    --am $cpt_dir \
    --device-id -1 \
    --channel -1 \
    --dict $cpt_dir/dict \
    --max-len 50 \
    --len-norm true
