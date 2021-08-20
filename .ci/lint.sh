#!/usr/bin/env bash

set -eu

sudo apt-get install -y shellcheck
cat requirements.txt | grep -E "flake8|cpplint" > requirements_lint.txt
pip install -r requirements_lint.txt

# for shell
shellcheck -e SC2086,SC2002 -x tests/python/*.sh utils/subword.sh scripts/*.sh
shellcheck -e SC2086,SC2001,SC2002,SC2010 -x examples/*/*/run.sh
# for python
flake8
# for c++
cpplint --recursive csrc
