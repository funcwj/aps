# Code Document

Providing a simple description of the design and code structure here. The main point of the APS design are:

* End-to-end training and evaluation. I give the implementation of the common feature transform used in ASR/SE/SS tasks with PyTorch in `aps.transform`. The stack of the transform layer can be used to extract the specific acoustic features. During training and evaluation, it serves as a sub-layer and can infer on CPU/GPU device with high performance. After collecting and formating the audio data, we can kick off training and evaluation without the need to prepare the feature and label beforehand.
For SE/SS tasks, we further perform data simulation in an online manner which can simulate large scale training data pairs on-the-fly, avoiding the time cost and disk usage in traditional offline simulation. With this fashion, we can easily realize "waveform in, unit out" style for ASR evaluation and "waveform in, waveform out" style for speech enhancement/separation training and evaluation. It's also convenient when exporting the model out and deploy them.

* Deal with different tasks and loss functions. In speech enhancement/separation, we often need to tune model using different objective functions, e.g., for the frequency domain model, we can adopt SiSNR (time domain) and MSE (frequency domain) for optimization. Instead of binding the loss function with the network itself, we use the `Task` class to handle the above issues. It inherits the class `torch.nn.Module` and accepts an arbitrary network as the parameter, but the forward function in different sub-class defines the computation of the loss function. In the training configurations, different tasks can be assigned for same network, e.g., to train a DCCRN model, we can choose one from `SnrTask`, `WaTask` and `LinearFreqSaTask`, etc. Decoupling the loss function and network prototype make it flexible to support new tasks & loss functions and also simplify the code of the network definition (we focus more on structures). Also as it provide unified interface, it's easy to manage different `Task` instances using one `Trainer`.


The relationship of the `aps.transform`, `aps.task`, `aps.trainer`, `aps.loader` and task dependent network structure (now defined in `aps.asr` and `aps.sse`) are shown below:
```
    ---------------------------------------------
    | -----------------------------             |
    | | (Transform) Network (...) |  Task (...) | => Trainer <= DataLoader
    | -----------------------------             |
    ---------------------------------------------
```

## `aps.distributed`

To deal with different distribution backend and unify the interface, e.g., Horovod, PyTorch.

## `aps.transform`

Customized feature transform module in APS. The feature transform layer from `aps.transform` is stacked as the feature extractor for ASR or enhancement/separation tasks, e.g., a stack of `SpectrogramTransform, AbsTransform, MelTransform, LogTransform` can used to extract log mel filter-bank features. Currently the supported feature transform layers are shown in the following:

* `RescaleTransform`
* `PreEmphasisTransform`
* `SpeedPerturbTransform`
* `SpectrogramTransform`
* `MagnitudeTransform`
* `TFTransposeTransform`
* `AbsTransform`
* `MelTransform`
* `LogTransform`
* `PowerTransform`
* `DiscreteCosineTransform`
* `CmvnTransform`
* `SpecAugTransform`
* `SpliceTransform`
* `DeltaTransform`
* `RefChannelTransform`
* `PhaseTransform`
* `IpdTransform`
* `DfTransform`
* `FixedBeamformer`

The instance of `AsrTransform` or `EnhTransform` is passed to network prototype defined in `aps.asr`/`aps.sse` as a parameter for user-specific feature extraction. The configurations should be provided in `.yaml` files.

## `aps.trainer`

Trainer in APS. Now the three Trainer instances are provided: Hovorod & PyTorch (`DistributedDataParallel`) & Apex (`DistributedDataParallel`) and all of them are inherited from the base class `aps.trainer.Trainer`:

* `DdpTrainer`: Single-GPU or multi-GPU training using PyTorch's DistributedDataParallel (default way and we disable the `DataParallel` as it's not as  efficient as DDP)
* `HvdTrainer`: Multi-GPU training using horovod (for someone who wants to use it.)
* `ApexTrainer`: Single-GPU or multi-GPU training using Apex's DistributedDataParallel, also aims to mixed precision training

The default trainer is `DdpTrainer` which could be used for both single/multi-GPU training. The scheduler of the learning rate and schedule sampling is defined in `aps.trainer.lr` and `aps.trainer.ss`. Refer the original code for details.

## `aps.task`

Supported task in APS. The `Task` class is responsible for the computation of an user-specific objective function, which is defined in the `forward()` function. Actually it's still a PyTorch's `Module` class and each one should inherit from `th.nn.Module`. Currently the supported task are shown below:

* `LmXentTask`: for LM training
* `CtcXentHybridTask`: CTC & Xent multi-task training for E2E ASR ([paper](https://arxiv.org/abs/1211.3711))
* `TransducerTask`: for RNNT training ([paper](https://ieeexplore.ieee.org/document/8068205))
* `MlEnhTask`: for unsupervised multi-channel speech enhancement ([paper](https://arxiv.org/abs/1904.01578))
* `SisnrTask`: SiSNR loss for time domain enhancement/separation model ([paper](https://arxiv.org/abs/1711.00541))
* `SnrTask`: SNR loss for time domain enhancement/separation model
* `WaTask`: waveform L1/L2 loss for time domain enhancement/separation model ([paper](https://arxiv.org/abs/1804.10204))
* `LinearFreqSaTask`:  spectral approximation loss for frequency domain enhancement/separation model ([paper](https://ieeexplore.ieee.org/document/8540037))
* `MelFreqSaTask`: mel domain spectral approximation loss for frequency domain enhancement/separation model
* `LinearTimeSaTask`: spectral approximation loss for time domain enhancement/separation model
* `MelTimeSaTask`: mel domain spectral approximation loss for time domain enhancement/separation model
* `ComplexMappingTask`: frequency domain complex mapping objective function ([paper](https://ieeexplore.ieee.org/document/9103053))
* `ComplexMaskingTask`: frequency domain complex mask training objective function
* `SseFreqKdTask`: frequency domain TS (teacher-student) learning task

## `aps.loader`

The supported data loader in APS. For acoustic model training, we have three options

* `am@raw`: Raw waveform data loader which do not need us to prepare acoustic features beforehead (recommended way).
* `am@kaldi`: The data loader that supports the Kaldi format feature.
* `am@command`: The dataloader which generates the training audio on-the-fly based on command-line configurations (see `se@command` for details).

For separation/enhancement model training, we also have two options

* `se@chunk`: Raw waveform data loader and also no need to prepare features.
* `se@command`: A data loader performing online data simulation which generates training audio pairs (noisy, single/multi-speaker, close-talk/far-field) on-the-fly based on the command-line configurations.
* `se@config`: Another version of data loader for online data simulation (based on json configurations)

For language model (target at ASR task), we have

* `lm@utt`: The utterance corpus data loader. We gather several utterances as one minibatch with neccessary padding.
* `lm@bptt`: The data loader used with BPTT training.

## `aps.distributed`

A package to handle distributed training and provide an unified interface. Now we only have two options: `torch` and `horovod`.

## `aps.asr`

The submodule for language model & acoustic model. Currently the implemented AM are:

* `AttASR`: Encoder/decoder AM with RNN decoders
* `EnhAttASR`: `AttASR` with multi-channel front-end
* `XfmrASR`: Encoder/decoder AM with transformer decoders
* `EnhXfmrASR`: `XfmrASR` with multi-channel front-end
* `TransducerASR`: RNNT AM with RNN decoders
* `XfmrTransducerASR`: RNNT AM with Transformer decoders

The transformer implementation is kept similar style with `torch.nn` package and I put them under `aps.asr.transformer` package. Now the `TransformerEncoder` supports the vanilla Transformer and [Conformer](https://arxiv.org/abs/2005.08100) with the following multi-head self-attention (MHSA):

* `ApsMultiheadAttention`: original MHSA proposed in "Attention is All You Need" ([paper](https://arxiv.org/abs/1706.03762))
* `RelMultiheadAttention`: MHSA using relative position representations proposed in "Self-Attention with Relative Position Representations" ([paper](https://arxiv.org/abs/1803.02155))
* `XlMultiheadAttention`: MHSA using relative position proposed in Transformer-XL ([paper](https://arxiv.org/abs/1901.02860))

Other variants of non-transformer encoder are provided in `aps.asr.base.encoder`:

* `PyTorchRNNEncoder`: stack of vanilla RNNs as encoder (using PyTorch's CUDNN backend)
* `VariantRNNEncoder`: RNNs with customized features
* `FSMNEncoder`: stack of FSMN as encoder
* `JitLSTMEncoder`: customized LSTM layers implemented with torch.jit
* `Conv1dEncoder`: stack of TDNN (conv1d) layers as encoder (can also be used for subsampling, following with other encoders).
* `Conv2dEncoder`: stack of conv2d layers (used for subsampling, following with other encoders)

`ConcatEncoder` is used for concatenation of different encoders, e.g., `conv1d + pytorch-rnn`, `conv2d + variant-rnn`.

and attention type:
* `DotAttention`: dot attention and its multi-head version `MHDotAttention` ([paper](https://arxiv.org/abs/1508.01211))
* `LocAttention`: location aware attention and its multi-head version `MHLocAttention` ([paper](https://arxiv.org/abs/1506.07503))
* `CtxAttention`: context attention and its multi-head version `MHCtxAttention` ([paper](https://arxiv.org/abs/1409.0473))

The decoders are much simple than encoders, now APS provides RNN and Transformer decoders for attention and transducer based AM, respectively:
* `aps.asr.base.decoder.PyTorchRNNDecoder`
* `aps.asr.transformer.decoder.TorchTransformerDecoder`
* `aps.asr.transducer.decoder.PyTorchRNNDecoder`
* `aps.asr.transducer.decoder.TorchTransformerDecoder`

The beam search algothrim is provided in `aps.asr.beam_search`.

## `aps.sse`

The submodule for speech enhancement/separation model. The provided model are shown below:

* `TimeTCN`: Time domain Conv-TasNet ([paper](https://arxiv.org/pdf/1809.07454.pdf))
* `FreqTCN`: Frequency domain TCN (Temporal Convolutional Network)
* `DCUNet`: Deep Complexed Unet
* `DCCRN`: Deep Complexed Convolutional Recurrent Network ([paper](https://arxiv.org/pdf/2008.00264.pdf))
* `DenseUnet`: A Unet structure network boosted with DenseBlock ([paper](https://arxiv.org/abs/2010.01703))
* `Phasen`: Phase and Harmonics Aware Speech Enhancement Network ([paper](https://arxiv.org/abs/1911.04697))
* `TimeDPRNN`: Time domain Dual-path RNN ([paper](https://arxiv.org/abs/1910.06379))
* `FreqDPRNN`: Frequency domain DPRNN
* `ToyRNN`: Basic RNN model
* `FreqRelXfmr`: A simple transformer model for enhancement/separation
* `DEMUCS`: Real Time Speech Enhancement in the Waveform Domain ([paper](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2409.pdf))
* `SepFormer`: Attention is All You Need in Speech Separation ([paper](https://arxiv.org/abs/2010.13154))
* `DFSMN`: Deep FSMN for speech enhancement

Note that in `aps.sse`, each network should have function `infer()` for inference while `forward()` is left for training only.

## `aps.streaming_asr`

The submodule for streaming ASR which is designed with TorchScript support and model deployment features.

## `aps.rt_sse`

The submodule for real time speech enhancement & separation which is also designed with TorchScript support and model deployment features. Refer to the demo code directory [aps/demos/real_time_enhancement](aps/demos/real_time_enhancement) for details.
