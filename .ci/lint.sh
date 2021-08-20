#!/usr/bin/env bash

set -eu

sudo apt-get install -y shellcheck

# for shell
shellcheck -e SC2086,SC2002 -x tests/*.sh utils/subword.sh scripts/*.sh
shellcheck -e SC2086,SC2001,SC2002,SC2010 -x examples/*/*/run.sh
# for python
pip install flake8 && flake8 --version && flake8
# for c++
cpplint --recursive csrc
