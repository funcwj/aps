#!/usr/bin/env bash

set -eu

[ $# -ne 2 ] && echo "Script format error: $0 <database-dir> <data-dir>" && exit 1

database_dir=$1 && mkdir -p $database_dir
data_dir=$2 && mkdir -p $data_dir

local/aidatatang_data_prep.sh $database_dir/aidatatang/aidatatang_200zh \
  $data_dir/aidatatang || exit 1;
local/aishell_data_prep.sh $database_dir/aishell/data_aishell \
  $data_dir/aishell || exit 1;
local/magicdata_data_prep.sh $database_dir/magicdata \
  $data_dir/magicdata || exit 1;
local/primewords_data_prep.sh $database_dir/primewords \
  $data_dir/primewords || exit 1;
local/stcmds_data_prep.sh $database_dir/stcmds $data_dir/stcmds || exit 1;
local/thchs30_data_prep.sh $database_dir/thchs/data_thchs30 \
  $data_dir/thchs || exit 1;
