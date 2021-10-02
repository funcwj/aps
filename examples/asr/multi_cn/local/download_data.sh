#!/usr/bin/env bash

set -eu

openslr_url=www.openslr.org

aidatatang_id=62
aishell_id=33
magicdata_id=68
primewords_id=47
stcmds_id=38
thchs_id=18

[ $# -ne 1 ] && echo "Script format error: $0 <database-dir>" && exit 1

database_dir=$1 && mkdir -p $database_dir

# training data
local/aidatatang_download_and_untar.sh $database_dir/aidatatang \
  $openslr_url/resources/$aidatatang_id aidatatang_200zh || exit 1;
local/aishell_download_and_untar.sh $database_dir/aishell \
  $openslr_url/resources/$aishell_id data_aishell || exit 1;
local/magicdata_download_and_untar.sh $database_dir/magicdata \
  $openslr_url/resources/$magicdata_id train_set || exit 1;
local/primewords_download_and_untar.sh $database_dir/primewords \
  $openslr_url/resources/$primewords_id || exit 1;
local/stcmds_download_and_untar.sh $database_dir/stcmds \
  $openslr_url/resources/$stcmds_id || exit 1;
local/thchs30_download_and_untar.sh $database_dir/thchs \
  $openslr_url/resources/$thchs_id data_thchs30 || exit 1;

# test data
local/thchs_download_and_untar.sh $database_dir/thchs \
  $openslr_url/resources/$thchs_id test-noise || exit 1;
local/magicdata_download_and_untar.sh $database_dir/magicdata \
  $openslr_url/resources/$magicdata_id dev_set || exit 1;
local/magicdata_download_and_untar.sh $database_dir/magicdata \
  $openslr_url/resources/$magicdata_id test_set || exit 1;
