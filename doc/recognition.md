## ASR - Quick Start

Training acoustic models.

### Data Preparation

We need to prepare data dependencies for training and cross-validation, e.g., to use raw waveform dataloader, `wav.scp`, `token` and `utt2dur` are needed. You can finish the job with the help of the [Kaldi](https://github.com/kaldi-asr/kaldi/egs) following the data preparation recipes.

1. Create directory `data/<dataset>/{train,valid}` and prepare `wav.scp`, `text` and `utt2dur` files.

    * `wav.scp`: single or multi-channel audio data 
    ```
    IC0001W0001	/.../aishell-2/wav/C0001/IC0001W0001.wav
    IC0001W0002	/.../aishell-2/wav/C0001/IC0001W0002.wav
    IC0001W0003	/.../aishell-2/wav/C0001/IC0001W0003.wav
    IC0001W0004	/.../aishell-2/wav/C0001/IC0001W0004.wav
    IC0001W0005	/.../aishell-2/wav/C0001/IC0001W0005.wav
    ...
    ```

    * `text`: transcription file
    ```
    IC0001W0001	厨房 用具
    IC0001W0002	电蒸锅
    IC0001W0003	电蒸锅
    IC0001W0004	嵌入式 灶具
    IC0001W0005	水槽 洗碗机
    ...
    ```

    * `utt2dur`: utterance duration (seconds) file
    ```
    IC0001W0001 2.027
    IC0001W0002 1.914
    IC0001W0003 1.914
    IC0001W0004 2.211
    IC0001W0005 2.375
    ...
    ```

2. Tokenizing the transcription `text` to `char` if needed:

    ```bash
    # 1)  add <space> between each character for English dataset
    cat /path/to/text | ./utils/tokenizer.pl --space "<space>" > /path/to/char
    # 2)  transform words to characters for Mandarin dataset
    cat /path/to/text | ./local/tokenizer.pl > /path/to/char
    ```

3. Prepare dictionary file (usually contains `<blank> <eos> <sos> <unk>`) in `data/<dataset>/dict`, which looks like:
    ```
    <blank> 0
    一 1
    丁 2
    七 3
    万 4
    丈 5
    三 6
    上 7
    下 8
    不 9
    ...
    ```
    
4. Encoding `text` or `char` to `token` file, replacing each unit with the ID in the dictionary, e.g.:
    ```
    IC0001W0001 577 1033 4682 2001
    IC0001W0002 833 5008 1393
    IC0001W0003 833 5008 1393
    IC0001W0004 2079 3358 3550 4870 2001
    IC0001W0005 3633 397 4144 3968 1661
    ...
    ```

### Training Configuration

Create .yaml configurations in `conf/<dataset>`, e.g., `conf/aishell_v2/1a.yaml` and configure the following keywords:

1. `nnet` & `nnet_conf`

    Now the supported AM (LM) are shown in [src/aps/asr/\_\_init\_\_.py](src/aps/asr/__init__.py):
    ```python
    nnet_cls = {
        # LM
        "rnn_lm": TorchRNNLM,
        "transformer_lm": TorchTransformerLM,
        # encoder-decoder structure, using RNN as decoder
        "att": AttASR,
        # AttASR with joint-mvdr front-end
        "mvdr_att": MvdrAttASR,
        # AttASR with fixed beamformer front-end
        "beam_att": BeamAttASR,
        # also encoder-decoder structure, but using transformer as decoder
        "transformer": TransformerASR,
        # TransformerASR with fixed beamformer front-end
        "beam_transformer": BeamTransformerASR,
        # TransformerASR with joint-mvdr front-end
        "mvdr_transformer": MvdrTransformerASR,
        # Transducer using Transformer as decoder
        "transformer_transducer": TransformerTransducerASR,
        # Transducer using RNN as decoder
        "common_transducer": TorchTransducerASR
    }
    ```
    Parameters definition for AM should be configured in `nnet_conf`, e.g., using `AttASR`:
    ```yaml
    nnet: "att"

    nnet_conf:
      input_size: 80
      # tdnn, fsmn, common, custom
      encoder_type: "tdnn"
      encoder_proj: 512
      encoder_kwargs: 
        tdnn_dim: 512
        tdnn_layers: 3
        tdnn_stride: "2,2,2"
        tdnn_dilation: "1,1,2"
        rnn: "lstm"
        rnn_layers: 3
        rnn_bidir: True
        rnn_dropout: 0.2
        rnn_hidden: 320
      decoder_dim: 512
      decoder_kwargs:
        dec_rnn: "lstm"
        rnn_layers: 2
        rnn_hidden: 512  # must eq decoder_dim
        rnn_dropout: 0
        input_feeding: True
        vocab_embeded: True
      att_type: "ctx"
      att_kwargs:
        att_dim: 512
    ```

2. `task` & `task_conf`

    The supported `Task` classes are shown in [src/aps/task/\_\_init\_\_.py](src/aps/task/__init__.py):
    ```python
    task_cls = {
        # for LM training
        "lm": LmXentTask,
        # for CTC & CE joint training
        "ctc_xent": CtcXentHybridTask,
        # for transducer training
        "transducer": TransducerTask,
        # for enhancement/separation task...
        "unsuper_enh": UnsuperEnhTask,
        "sisnr": SisnrTask,
        "spectra_appro": SaTask
    }
    ```
    Parameters for selected `Task` should be configured in `task_conf`, e.g., using `CtcXentHybridTask` for `AttASR`:
    ```yaml
    task: "ctc_xent"

    task_conf:
      lsm_factor: 0.1
      ctc_regularization: 0.2
    ```

3. `asr_transform` & `enh_transform`

    Parameters of the feature tranformation (extraction) for ASR or enhancement front-end, defined in [src/aps/feats/asr.py](src/aps/feats/asr.py) and [src/aps/feats/asr.py](src/aps/feats/asr.py), e.g., for ASR task, to extract log mel-fbank features with spec-augumentation:
    ```yaml
    asr_transform:
        feats: "fbank-log-cmvn-aug"
        frame_len: 400
        frame_hop: 160
        window: "hamm"
        round_pow_of_two: True
        sr: 16000
        num_mels: 80
        norm_mean: True
        norm_var: True
        aug_prob: 0.5
    ```
    and STFT features with cosIPD:
    ```yaml
    enh_transform:
        feats: "spectrogram-log-cmvn-ipd"
        frame_len: 400
        frame_hop: 160
        window: "hann"
        norm_mean: True
        norm_var: True
        ipd_index: "1,0;2,0;3,0"
        cos_ipd: True
    ```

4. `data_conf`

    Training and cross-validation data configurations, e.g., using raw waveform dataloader:
    ```yaml
    data_conf:
        fmt: "wav"  # or "kaldi", "conf"
        loader:
            max_token_num: 400
            adapt_token_num: 150
            max_dur: 30 # (s)
            min_dur: 0.4 # (s)
            adapt_dur: 10 # (s)
        train:
            wav_scp: "data/aishell_v2/train/wav.scp"
            utt2dur: "data/aishell_v2/train/utt2dur"
            token: "data/aishell_v2/train/token"
        valid:
            wav_scp: "data/aishell_v2/dev/wav.scp"
            utt2dur: "data/aishell_v2/dev/utt2dur"
            token: "data/aishell_v2/dev/token"
    ```
    The supported dataloader for AM training is defined in [src/asr/loader/am](src/asr/loader/am).

5. `trainer_conf`

    Network training configurations, e.g., training `AttASR` with linear schedule sampling strategy:
    ```yaml
    trainer_conf:
        # optimizer
        optimizer: "adam"
        optimizer_kwargs:
            lr: 1.0e-3
            weight_decay: 1.0e-5
        lr_scheduler_kwargs:
            min_lr: 1.0e-8
            patience: 1
            factor: 0.5
        # scheduler sampling
        ss_scheduler: "linear"
        ss_scheduler_kwargs:
            ssr: 0.3
            epoch_beg: 10
            epoch_end: 26
            update_interval: 4 
        no_impr: 6 # early stop
        no_impr_thres: 0.2
        lsm_factor: 0.1     # label smooth factor
        clip_gradient: 5    # gradient clipping
        ctc_regularization: 0.2 # CTC factor
        stop_criterion: "accu"
    ```

### Training and Decoding

Now we have data directory `data/<dataset>` and training configurations `conf/<dataset>/1a.yaml`. Using scripts in folder [scripts](scripts) to start training and decoding

1. Training

    ```bash
    # single GPU
    ./train_am.sh --batch-size 64 --num-workers 4 --seed 666 <dataset> 1a
    # multiple GPUs on single node
    ./distributed_train_am.sh --batch-size 64 --num-workers 2 --num-process 2 --seed 666 <dataset> 1a
    ```
    This will create checkpoint directory in `exp/<dataset>/1a`

2. Decoding (beam search)

    ```bash
    # GPU
    ./decode.sh --max-len 30 --nbest 8 --beam-size 24 --gpu 0 --dict data/<dataset>/dict \
        <dataset> 1a <tst-wav-scp> exp/<dataset>/1a/tst
    # CPU: parallel mode
    ./decode_parallel.sh --nj 40 --cmd "utils/run.pl" --max-len 30 --nbest 8 \
        --beam-size 24 --gpu 0 --dict data/<dataset>/dict <dataset> 1a \
        <tst-wav-scp> exp/<dataset>/1a/tst
    ```
    This will dump decoding transcripts and n-best hypothesis in `exp/<dataset>/1a/tst`

3. WER evaluation
    
    ```bash
    ./src/computer_wer.py exp/<dataset>/1a/tst/<decoding-transcription> <tst-transcription>
    ```
