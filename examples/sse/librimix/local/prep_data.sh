#!/usr/bin/env bash

set -eu

[ $# -ne 1 ] && echo "$0: script format error: <librimix_wav_dir> <local_data_dir>" && exit 0

librimix_wav_dir=$1
data_dir=$2
