# Usage overview

My suggested working directory looks like:
```bash
bin conf data exp scripts utils 
```
and it can be initialzied using `scripts/init_workspace.sh`. For example:
```bash
export APS_ROOT=/path/to/aps
$APS_ROOT/scripts/init_workspace.sh asr1
$APS_ROOT/scripts/init_workspace.sh sep1
$APS_ROOT/scripts/init_workspace.sh enh1
```
will make directory current `workspace` like (`APS_ROOT=../aps`):
```
.
├── bin -> ../aps/bin
├── conf
│   ├── asr1
│   ├── enh1
│   └── sep1
├── data
│   ├── asr1
│   ├── enh1
│   └── sep1
├── scripts -> ../aps/scripts
└── utils -> ../aps/utils
```

Assuming that we've prepared training & test data and experiment configurations under `data` and `conf` directory, the model training can be easily started by running the provided scripts under `scripts`:

* `scripts/train_{am,lm,ss}.py`: Single-GPU training for acoustic model, language model and speech enhancement/separation model, respectively.
* `scripts/distributed_train_{am,ss}.py`: Distributed training (currently single-node multi-GPU) for acoustic model and speech enhancement/separation model.

E.g., running
```bash
./scripts/train_am.sh --batch-size 32 --gpu 0 asr1 1a
```
will load .yaml configuration from `conf/asr1/1a.yaml` and create checkpoint directory in `exp/asr1/1a`. After one epoch is done, the directory looks like:
```
.
├── bin -> ../aps/bin
├── conf
│   ├── asr1
│   │   └── 1a.yaml
│   ├── enh1
│   └── sep1
├── data
│   ├── asr1
│   │   ├── dev
│   │   │   ├── text
│   │   │   ├── token
│   │   │   ├── utt2dur
│   │   │   └── wav.scp
│   │   ├── train
│   │   │   ├── text
│   │   │   ├── token
│   │   │   ├── utt2dur
│   │   │   └── wav.scp
│   │   └── tst
│   │       ├── text
│   │       └── wav.scp
│   ├── enh1
│   └── sep1
├── exp
│   └── asr1
│       └── 1a
│           ├── best.pt.tar
│           ├── last.pt.tar
│           ├── train.yaml
│           └── trainer.log
├── scripts -> ../aps/scripts
└── utils -> ../aps/utils
```
(the data directory `data/asr1/{train,dev,tst}` is a typical setup for acoustic model training)

After the end of the model training, we can start task dependent evaluation using the assets under checkpoint directory. Some recipes are available under `aps/egs`.
