#!/usr/bin/env bash

set -eu

[ $# -ne 1 ] && echo "$0: script format error: <librimix_data_dir>"

librimix_data_dir=$1

mkdir -p exp && cd exp && git clone https://github.com/JorisCos/LibriMix.git

cd LibriMix && bash generate_librimix.sh $librimix_data_dir
