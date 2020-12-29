# Usage overview

There is no `setup.py` script for installing the APS package in the repository and actually I don't suggest readers doing that. The following shows the recommended way to use the APS.

A typical working directory looks like:
```bash
cmd conf data exp scripts utils
```
It tracks several experiments and could be initialized using `scripts/init_workspace.sh`, e.g.,
```bash
export APS_ROOT=/path/to/aps
$APS_ROOT/scripts/init_workspace.sh wsj0_2mix
$APS_ROOT/scripts/init_workspace.sh aishell_v1
```
will make directory current `workspace` like (`APS_ROOT=../aps`):
```
.
├── cmd -> ../aps/cmd
├── conf
│   ├── wsj0_2mix
│   └── aishell_v1
├── data
│   ├── wsj0_2mix
│   └── aishell_v1
├── scripts -> ../aps/scripts
└── utils -> ../aps/utils
```

Assuming that we've prepared data and experiment configurations under directory `data` and `conf` (see [Instruction](instruction.md) for details), the model training can be easily kicked off by running the provided scripts under [scripts](../scripts):

* `scripts/train.sh`: Single-GPU training for acoustic model (AM), language model (LM) and speech separation/enhancement (SSE) model, respectively.
* `scripts/distributed_train.sh`: Distributed training (currently single-node multi-GPU) for AM & SSE models.

E.g., running
```bash
./scripts/train_am.sh --batch-size 32 --gpu 0 aishell_v1 1a
```
will load `.yaml` configuration from `conf/aishell_v1/1a.yaml` and create checkpoint directory in `exp/aishell_v1/1a` (The feature extraction is performed on GPU so there is no need to prepare features for all the tasks). After one epoch is done, the directory looks like:
```
.
├── cmd -> ../aps/cmd
├── conf
│   ├── aishell_v1
│   │   └── 1a.yaml
│   └── wsj0_2mix
├── data
│   ├── aishell_v1
│   │   ├── dev
│   │   │   ├── text
│   │   │   ├── utt2dur
│   │   │   └── wav.scp
│   │   ├── train
│   │   │   ├── text
│   │   │   ├── utt2dur
│   │   │   └── wav.scp
│   │   └── tst
│   │       ├── text
│   │       └── wav.scp
│   └── wsj0_2mix
├── exp
│   └── aishell_v1
│       └── 1a
│           ├── best.pt.tar
│           ├── last.pt.tar
│           ├── train.yaml
│           └── trainer.log
├── scripts -> ../aps/scripts
└── utils -> ../aps/utils
```
(the data directory `data/asr1/{train,dev,tst}` is a typical setup for AM training)

After the end of the model training, we can start task dependent evaluation using the assets under checkpoint directory. Some recipes are available under [aps/examples](../examples).
