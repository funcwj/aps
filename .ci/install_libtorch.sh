#!/usr/bin/env bash

set -eu

version=1.7.1
name=libtorch-shared-with-deps-$version
url=https://download.pytorch.org/libtorch/cpu/$name%2Bcpu.zip

mkdir -p third_party && cd third_party
wget $url && unzip $name+cpu.zip && rm $name+cpu.zip
