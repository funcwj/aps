## Speech Separation

Training blind speech separation (BSS) models.

The networks for blind speech enhancement tasks are implemented under `src/aps/sep/bss`.

### Data Preparation

Creating data directory `data/sep_egs/{train,dev,tst}` and preparing data scripts `{mix,spk1,spk2}.scp` for each subset (i.e., `train`, `dev`, `tst`). Now we get
```shell
jwu$ ls data/sep_egs/train/
mix.scp  spk1.scp  spk2.scp
jwu$ ls data/sep_egs/dev/
mix.scp  spk1.scp  spk2.scp
jwu$ ls data/sep_egs/tst/
mix.scp  spk1.scp  spk2.scp
```

### Training Configuration

Create .yaml configuration file under `conf/enh_egs`, e.g., `conf/sep_egs/1a.yaml` using the above `sep_egs` dataset

1. `nnet`

    Now the supported BSS models are defined in [src/aps/sep/bss](src/aps/sep/bss):
    * Conv-TasNet (`tasnet`)
    * DPRNN (`dprnn`)
    * FreqDomainToyRNN (`freq_toy`) & TimeDomainToyRNN (`time_toy`)

2. `nnet_conf`

    Parameters configuration for SE models defined in [src/aps/sep/bss](src/aps/sep/bss), e.g., `time_toy`:

    ```yaml
    # network parameter (see src/aps/sep/enh/*.py or src/aps/sep/*.py)
    nnet_conf:
      num_spks: 2
      input_size: 257
      num_bins: 257
      rnn: "lstm"
      rnn_layers: 3
      rnn_hidden: 512
      rnn_dropout: 0.2
      rnn_bidir: True
      output_nonlinear: "relu"
    ```

3. `task` and `task_conf`

    The `Task` classes for speech enhancement/separation are defined in [src/aps/task/sep.py](src/aps/task/sep.py). We adopt `sisnr` task here
    to match the `time_toy` model:
    ```yaml
    # task name (using si-snr as objective function)
    task: sisnr
    # task parameter for BSS (see src/aps/task/sep.py)
    task_conf:
      num_spks: 2
      permute: true
    ```
    To use spectrum approximation (SA) objective function instead of Si-SNR, using task `spectra_appro`
    ```yaml
    task: spectra_appro
    task_conf:
      num_spks: 2
      permute: true
      truncated: 1
      phase_sensitive: true
    ```
    and frequency-domain models, e.g.,
    ```yaml
    nnet: freq_toy
    nnet_conf:
      # same as time_toy
      ...
    ```

4. `enh_transform`

    Parameters of the feature extraction for speech enhancement/separation tasks, defined in [src/aps/feats/enh.py](src/aps/feats/enh.py), e.g., to extract log spectrogram features for `time_toy` model
    ```yaml
    # feature transform for SE (see src/aps/feats/enh.py)
    enh_transform:
      feats: "spectrogram-log-cmvn"
      frame_len: 512
      frame_hop: 256
      window: "sqrthann"
      round_pow_of_two: true
    ```

5. `data_conf`

    Training and cross-validation data configuration, e.g., for dataset in `data/enh_egs`:
    ```yaml
    # dataset configuration (see src/aps/loader/ss/chunk.py)
    data_conf:
      fmt: "enh"
      loader:
        sr: 16000
        chunk_size: 64000
      train:
        mix_scp: "data/sep_egs/train/mix.scp"
        ref_scp: "data/sep_egs/train/spk1.scp,data/sep_egs/train/spk2.scp"
      valid:
        mix_scp: "data/sep_egs/dev/mix.scp"
        ref_scp: "data/sep_egs/dev/spk1.scp,data/sep_egs/dev/spk2.scp"
    ```

6. `trainer_conf`

    Network training configuration. 
    ```yaml
    # trainer configurations (see src/aps/trainer/ddp.py)
    trainer_conf:
      optimizer: "adam"
      optimizer_kwargs:
        lr: 1.0e-3
        weight_decay: 1.0e-5
      lr_scheduler_kwargs:
        min_lr: 1.0e-8
        patience: 1
        factor: 0.5
      no_impr: 4
      no_impr_thres: 0.2
      stop_criterion: "loss"
      clip_gradient: 5
    ```

### Training and inference

```bash
# training BSS model using conf/sep_egs/1a.yaml
./scripts/train_ss.sh \
  --batch-size 16 \
  --gpu 0 \
  --num-workers 4 \
  --seed 666 \
  --epoches 50 \
  sep_egs 1a &
# inference (in GPU-0)
./src/blind_separate.py \
  --gpu 0 \
  --checkpoint exp/sep_egs/1a \
  data/sep_egs/tst/mix.scp \
  data/sep_egs/tst/bss
```


