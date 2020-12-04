# APS

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/funcwj/aps)
[![Testing](https://github.com/funcwj/aps/workflows/Unit%20Testing/badge.svg)](https://github.com/funcwj/aps/workflows/Unit%20Testing/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A workspace for speech related tasks, including single/multi-channel speech enhancement & separation & recognition. The goal is to make it easy and flexible to conduct experiments, verify ideas, implement and evaluate neural network based methods.

## Setup

```shell
git clone https://github.com/funcwj/aps
# set up the python environments
# 1) using "pip install -r requirements.txt" or
# 2) create conda enviroments based on requirements.txt (see docker/*.Dockerfile)
cd aps && pip install -r requirements.txt
```
For developers (who want to make commits or PRs), please install the `git-lfs` first (as I use Git LFS to track some PyTorch checkpoints for testing, see [link](https://github.com/git-lfs/git-lfs/wiki/Installation) as the reference) and contiune running
```shell
# set up the git hook scripts
pre-commit install
# set up git-lfs
git lfs install
# make sure the http version used by git is v1.1 or the pushing of the code may fail
git config http.version HTTP/1.1
```
to setup the development environments.

## Introduction

* [Overivew](docs/overview.md)
* [Structure](docs/code.md)
* [Instruction](docs/instruction.md)
* [Q & A](docs/qa.md)
* [Examples](examples)

## Acknowledge

The project was started at 2019.03 when the author was a master student of the Audio, Speech and Language Processing Group (ASLP) in Northwestern Polytechnical University (NWPU), Xi'an, China.
