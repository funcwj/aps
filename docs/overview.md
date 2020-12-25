# Usage overview

There is no `setup.py` script for installing the package in the repository and actually I don't suggest readers doing that. The following shows the recommended way to use the APS package.

A typical working directory looks like:
```bash
cmd conf data exp scripts utils
```
and it can be initialized using `scripts/init_workspace.sh`, e.g.,
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

Assuming that we've prepared training & test data and experiment configurations under `data` and `conf` directory (see [Instruction](instruction.md) for details), the model training can be easily started by running the provided scripts under [scripts](../scripts):

* `scripts/train_{am,lm,ss}.sh`: Single-GPU training for acoustic model, language model and speech enhancement/separation model, respectively.
* `scripts/distributed_train_{am,ss}.sh`: Distributed training (currently single-node multi-GPU) for acoustic model and speech enhancement/separation model.

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
(the data directory `data/asr1/{train,dev,tst}` is a typical setup for acoustic model training)

After the end of the model training, we can start task dependent evaluation using the assets under checkpoint directory. Some recipes are available under [aps/examples](../examples).
