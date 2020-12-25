# APS

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/funcwj/aps)
[![Testing](https://github.com/funcwj/aps/workflows/Unit%20Testing/badge.svg)](https://github.com/funcwj/aps/workflows/Unit%20Testing/badge.svg)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

My workspace for speech related tasks, e.g., single/multi-channel speech enhancement & separation & recognition. The goal is to simplify the training and evaluation procedure and make it easy and flexible to do experiments and verify neural network based methods.

## Setup

```shell
git clone https://github.com/funcwj/aps
# set up the python environments
# 1) using "pip install -r requirements.txt" or
# 2) create conda enviroments based on requirements.txt (see docker/*.Dockerfile)
# horovod is optional if you don't want to use that.
cd aps && pip install -r requirements.txt
```
For developers (who want to make commits or PRs), contiune running
```shell
# set up the git hook scripts
pre-commit install
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
