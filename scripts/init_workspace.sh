#!/usr/bin/env bash

# Copyright 2020 Jian Wu
# License: Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

[ $# -ne 1 ] && echo "$0: Script format: <name>" && exit 1

[ -z $APS_ROOT ] && echo "$0: export APS_ROOT=/path/to/aps first" && exit 1

name=$1

mkdir -p {conf,data}/$name

[ ! -s bin ] && ln -s $APS_ROOT/bin
[ ! -s utils ] && ln -s $APS_ROOT/utils
[ ! -s scripts ] && ln -s $APS_ROOT/scripts

echo "$0: init workspace for dataset $name done"
