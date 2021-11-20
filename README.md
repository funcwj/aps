# APS

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/funcwj/aps)
[![codecov](https://codecov.io/gh/funcwj/aps/branch/master/graph/badge.svg)](https://codecov.io/gh/funcwj/aps)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/funcwj/aps/actions/workflows/test_python.yml/badge.svg)](https://github.com/funcwj/aps/actions/workflows/test_python.yml/badge.svg)

My open-source toolkit for speech related tasks, e.g., single/multi-channel speech enhancement & separation & recognition. The goal is to simplify the training and evaluation procedure and make it easy and flexible for me to do experiments and verify neural network based methods.

## Setup

```shell
git clone https://github.com/funcwj/aps
# set up the python environments
# 1) using "pip install -r requirements.txt" or
# 2) create conda enviroments based on requirements.txt (recommended, see docker/Dockerfile)
cd aps && pip install -r requirements.txt # the optional packages are not shown in requirements.txt
```
For developers (who want to make commits or PRs), continue to run
```shell
# set up the git hook scripts
pip install -r requirements-dev.txt && pre-commit install
```
to setup the development environments. To build C++ sources and demo commands, running
```shell
mkdir build && cd build
cmake .. && make -j
```

## Introduction

* [Feature](docs/feature.md)
* [Overivew](docs/overview.md)
* [Structure](docs/code.md)
* [Instruction](docs/instruction.md)
* [Q & A](docs/qa.md)
* [Issues](docs/issue.md)
* [Examples](examples)

## Acknowledge

The project was started at early 2019 when the author was a master student of the [Audio, Speech and Language Processing Group](http://www.npu-aslp.org) (ASLP) in [Northwestern Polytechnical University](https://www.nwpu.edu.cn) (NWPU), Xi'an, China. Originally it was used to collect the source code of the experiments that the author did in the past.
