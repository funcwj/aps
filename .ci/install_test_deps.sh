#!/usr/bin/env bash

set -eu

nj=$(nproc)
work_dir=$PWD
sudo apt-get update
sudo apt-get install -y libsndfile-dev ffmpeg curl
# git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
# python dependencies
pip install --upgrade pip wheel
cat < requirements.txt | grep -v -E "warp_rnnt|horovod|kenlm" > requirements_cpu.txt && pip install -r requirements_cpu.txt
pip install numba==0.48
# RNNT (can run on CPU)
git clone https://github.com/HawkAaron/warp-transducer.git && cd warp-transducer
mkdir build && cd build && cmake .. && make -j $nj && sudo make install && sudo ldconfig -v
cd ../pytorch_binding && python setup.py install && cd $work_dir && rm -rf warp-transducer
# sentencepiece
git clone https://github.com/google/sentencepiece.git && cd sentencepiece
mkdir build && cd build && cmake .. && make -j $nj
sudo make install && sudo ldconfig -v && cd $work_dir && rm -rf sentencepiece
