#!/usr/bin/env bash

# Copyright 2021 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eu
stage="1-4"

database_dir=/home/jwu/doc/data/openslr_cn
data_dir=data/multi_cn
test_sets="aishell aidatatang magicdata thchs"

. ./utils/parse_options.sh || exit 1

beg=$(echo $stage | awk -F '-' '{print $1}')
end=$(echo $stage | awk -F '-' '{print $2}')
[ -z $end ] && end=$beg

if [ $end -ge 1 ] && [ $beg -le 1 ]; then
  echo "Stage 1: downloading & preparing the data ..."
  # local/download_data.sh $database_dir
  # local/prepare_data.sh $database_dir $data_dir
  local/format_data.sh $data_dir
fi
