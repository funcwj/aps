#!/usr/bin/env bash

set -eu

# run bash .ci/install_csrc_deps.sh
# install packages to directory aps/third_party

[ ! -d third_party ] && mkdir -p third_party

cd third_party

# libtorch
version=1.8.0
name=libtorch-shared-with-deps-$version
url=https://download.pytorch.org/libtorch/cpu/$name%2Bcpu.zip
wget $url && unzip $name+cpu.zip && rm $name+cpu.zip && mv libtorch/{bin,include,lib} . && rm -rf libtorch
