#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu

stage=1
librimix_data_dir=/mnt/jwu/librimix

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: simulate data ..."
  local/simu_librimix.sh $librimix_data_dir
fi
