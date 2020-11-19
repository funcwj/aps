#!/usr/bin/env bash

set -eu

# Pyhon dependencies
sudo apt-get install -y libsndfile-dev ffmpeg
pip install --upgrade pip wheel
cat < requirements.txt | grep -v -E "warp_rnnt|horovod|kenlm" > requirements_cpu.txt && pip install -r requirements_cpu.txt
pip install numba==0.48
# RNNT (can run on CPU)
git clone https://github.com/HawkAaron/warp-transducer.git && cd warp-transducer
mkdir build && cd build && cmake .. && make
cd ../pytorch_binding && python setup.py install
