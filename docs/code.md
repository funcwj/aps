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

## `aps.transform`

Customized feature transform module in APS. The feature transform layer from `aps.transform` is stacked as the feature extractor for ASR or enhancement/separation tasks, e.g., a stack of `SpectrogramTransform, AbsTransform, MelTransform, LogTransform` can used to extract log mel filter-bank features. Currently the supported feature transform layers are shown in the following:

* `SpeedPerturbTransform`
* `PreEmphasisTransform`
* `SpectrogramTransform`
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
* `CtcXentHybridTask`: CTC & Xent multi-task training for E2E ASR
* `TransducerTask`: for RNNT training
* `UnsuperEnhTask`: for unsupervised multi-channel speech enhancement
* `SisnrTask`: SiSNR loss for time domain enhancement/separation model
* `SnrTask`: SNR loss for time domain enhancement/separation model
* `WaTask`: waveform L1/L2 loss for time domain enhancement/separation model
* `LinearFreqSaTask`:  spectral approximation loss for frequency domain enhancement/separation model
* `MelFreqSaTask`: mel domain spectral approximation loss for frequency domain enhancement/separation model
* `LinearFreqSaTask`: spectral approximation loss for time domain enhancement/separation model
* `MelTimeSaTask`: mel domain spectral approximation loss for time domain enhancement/separation model
* `ComplexMappingTask`: frequency domain complex mapping objective function

## `aps.loader`

The supported data loader in APS. For acoustic model training, we have three options

* `am_raw`: Raw waveform data loader which do not need us to prepare acoustic features beforehead (recommended way).
* `am_kaldi`: The data loader that supports the Kaldi format feature.
* `am_online`: The dataloader which generates the training audio (noisy, far-field, etc) on-the-fly.

For separation/enhancement model training, we also have two options

* `se_chunk`: Raw waveform data loader and also no need to prepare features.
* `se_online`: A data loader performing online data simulation which generates training audio pairs (noisy, single/multi-speaker, close-talk/far-field) on-the-fly.

For language model (target at ASR task), we have

* `lm_utt`: The utterance corpus data loader.

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

The transformer implementation is kept similar style with `torch.nn` package and I put them under `aps.asr.xfmr` package. Now we have four encoders:

* `TorchTransformerEncoder`: encoder that uses transformer encoder provided in `torch.nn`
* `RelTransformerEncoder`: transformer encoder with relative position encodings
* `RelXLTransformerEncoder`: transformer-xl encoder
* `ConformerEncoder`: convolution augmented transformer encoder

Other variants of non-transformer encoder are provided in `aps.asr.base.encoder`:

* `PyTorchRNNEncoder`: stack of vanilla RNNs as encoder (using PyTorch's CUDNN backend)
* `VariantRNNEncoder`: RNNs with customized features
* `FSMNEncoder`: stack of FSMN as encoder
* `JitLSTMEncoder`: customized LSTM layers implemented with torch.jit
* `Conv1dEncoder`: stack of TDNN (conv1d) layers as encoder (can also be used for subsampling, following with other encoders).
* `Conv2dEncoder`: stack of conv2d layers (used for subsampling, following with other encoders)

`ConcatEncoder` is used for concatenation of different encoders, e.g., `conv1d + pytorch-rnn`, `conv2d + variant-rnn`.

and attention type:
* `DotAttention`: dot attention and its multi-head version `MHDotAttention`
* `LocAttention`: location aware attention and its multi-head version `MHLocAttention`
* `CtxAttention`: context attention and its multi-head version `MHCtxAttention`

The decoders are much simple than encoders, now APS provides RNN and Transformer decoders for attention and transducer based AM, respectively:
* `aps.asr.base.decoder.PyTorchRNNDecoder`
* `aps.asr.xfmr.decoder.TorchTransformerDecoder`
* `aps.asr.transducer.decoder.PyTorchRNNDecoder`
* `aps.asr.transducer.decoder.TorchTransformerDecoder`

The beam search algothrim is provided in `aps.asr.beam_search`.

## `aps.sse`

The submodule for speech enhancement/separation model. The provided model are shown below:

* `TimeConvTasNet`: Time domain [Conv-TasNet](https://arxiv.org/pdf/1809.07454.pdf)
* `FreqConvTasNet`: Frequency domain TCN (Temporal Convolutional Network)
* `DCUNet`: Deep Complexed Unet
* `DCCRN`: [Deep Complexed Convolutional Recurrent Network](https://arxiv.org/pdf/2008.00264.pdf)
* `DenseUnet`: [Unet boosted using DenseBlocks](https://arxiv.org/abs/2010.01703)
* `Phasen`: [Phasen network](https://arxiv.org/abs/1911.04697)
* `CRNet`: Convolutional Recurrent Network for speech enhancement
* `TimeDPRNN`: Time domain [DPRNN](https://arxiv.org/abs/1910.06379)
* `FreqDPRNN`: Frequency domain DPRNN
* `ToyRNN`: Basic RNN model
* `FreqRelXfmr`: A simple transformer model for enhancement/separation

Note that in `aps.sse`, each network should have function `infer()` for inference while `forward()` is left for training only.
