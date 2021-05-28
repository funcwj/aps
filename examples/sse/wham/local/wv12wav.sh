#!/usr/bin/env bash

set -eu

[ $# -ne 1 ] && echo "Script format error: $0 <wsj0-dir>" && exit 1

wsj0_dir=$1
cur_dir=$PWD
sph2pipe=sph2pipe

if [ ! `which sph2pipe` ]; then
  echo "Could not find sph2pipe, install it first..."
  wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
  tar -zxf sph2pipe_v2.5.tar.gz && cd sph2pipe_v2.5
  gcc -o sph2pipe *.c -lm && cd .. && rm -rf sph2pipe_v2.5.tar.gz
  sph2pipe=$PWD/sph2pipe_v2.5/sph2pipe
  cd $cur_dir
fi

echo "$0: convert *.wv1 => *.wav ..."
find $wsj0_dir -name "*.wv1" | awk '{print $1" "$1}' | sed 's:wv1:wav:' | \
  awk -v cmd=$sph2pipe '{printf("%s -f wav %s %s\n", cmd, $2, $1)}' | bash

rm -rf sph2pipe_v2.5
echo "$0: done"
