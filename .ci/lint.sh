#!/usr/bin/env bash

set -eu

sudo apt-get install -y shellcheck
shellcheck -e SC2086,SC2002 -x tests/*.sh utils/subword.sh scripts/*.sh
shellcheck -e SC2086,SC2001,SC2002,SC2010 -x examples/*/*/run.sh

pip install flake8 && flake8 --version && flake8
