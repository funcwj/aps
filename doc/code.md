# Code Document

Providing a simple description of the code structure here.

## `aps.transform`

Customized feature transform module in aps. The feature transform layer from `aps.transform` is stacked as the feature extractor for ASR or enhancement/separation tasks, e.g., a stack of `SpectrogramTransform-AbsTransform-MelTransform-LogTransform` can used to extract log mel-filter bank features. Currently the supported feature transform layers are shown in the following:

* `SpectrogramTransform`
* `AbsTransform`
* `MelTransform`
* `LogTransform`
* `DiscreteCosineTransform`
* `CmvnTransform`
* `SpecAugTransform`
* `SpliceTransform`
* `DeltaTransform`
* `IpdTransform`
* `DfTransform`
* `FixedBeamformer`

The instance of `AsrTransform` or `EnhTransform` is passed to network prototype defined in `aps.asr`/`aps.sse` as a parameter for user-specific feature extraction.

## `aps.trainer`

Trainer module in aps. Now we use `horovod` & `torch.nn.parallel.DistributedDataParallel` for distributed training.

* `HvdTrainer`: Multi-GPU training using horovod
* `DdpTrainer`: Single-GPU or multi-GPU training using PyTorch's DistributedDataParallel
* `ApexTrainer`: Single-GPU or multi-GPU training using Apex's DistributedDataParallel, also aims to mixed precision training

The learning rate and schedule sampling scheduler is defined in `aps.trainer.lr` and `aps.trainer.ss`.

## `aps.task`

Supported task in aps. The `Task` class is responsible for the computation of an user-specific objective function, which is defined in the `forward()` function. The supported task are shown below:

* `LmXentTask`: for LM training
* `CtcXentHybridTask`: for CTC & Attention based AM training
* `TransducerTask`: for RNNT training
* `UnsuperEnhTask`: for unsupervised enhancement training
* `SisnrTask`: SiSNR loss for time domain enhancement/separation model
* `SnrTask`: SNR loss for time domain enhancement/separation model
* `WaTask`: waveform L1/L2 loss for time domain enhancement/separation model
* `LinearFreqSaTask`:  spectral approximation loss for frequency domain enhancement/separation model
* `MelFreqSaTask`: mel domain spectral approximation loss for frequency domain enhancement/separation model
* `LinearFreqSaTask`: spectral approximation loss for time domain enhancement/separation model
* `MelTimeSaTask`: mel domain spectral approximation loss for time domain enhancement/separation model

## `aps.loader`

Supported data loader in aps. For acoustic model training, we have

* `am_wav`: Raw waveform data loader which do not need us to prepare acoustic features beforehead (recommended).
* `am_kaldi`: Data loader that supports feature format in Kaldi toolkit.

For enhancement/separation model, we have

* `ss_chunk`: Raw waveform data loader and also no need to prepare features.
* `ss_online`: Online data loader which generate training audio (noisy, single/multi-speaker, close-talk/far-field) on-the-fly.

## `aps.distributed`

A package to handle distributed training and provide an unified interface.

## `aps.asr`

Language model & E2E acoustic model. The implemented AM are:

* `AttASR`: Attention based encoder-decoder AM with RNN based decoder
* `EnhAttASR`: `AttASR` with multi-channel front-end
* `TransformerASR`: Attention based encoder-decoder AM with tran sformer as decoder
* `EnhTransformerASR`: `TransformerASR` with multi-channel front-end
* `TorchTransducerASR`: RNNT AM with RNN based decoder
* `TransformerTransducerASR`: RNNT AM with Transformer based decoder

The transformer implementation comes from `torch.nn` package and each network should have function `beam_search()` for decoding. Variants of encoder are provided in `aps.asr.base.encoder`:

* `TorchRNNEncoder`: stack of RNNs as encoder
* `CustomRNNEncoder`: RNNs with customized features
* `TDNNEncoder`: stack of TDNN as encoder
* `FSMNEncoder`: stack of FSMN as encoder
* `TimeDelayRNNEncoder`: TDNN (for sub-sampling) + RNNs as encoder
* `TimeDelayFSMNEncoder`: TDNN (for sub-sampling) + FSMN as encoder

and attention type:
* `DotAttention`: dot attention and its multi-head version `MHDotAttention`
* `LocAttention`: location aware attention and its multi-head version `MHLocAttention`
* `CtxAttention`: context attention and its multi-head version `MHCtxAttention`

## `aps.sep`

Speech enhancement/separation model. The implemented model are shown below:

* `TimeConvTasNet`: Time domain Conv-TasNet
* `FreqConvTasNet`: Frequency domain TCN (Temporal Convolutional Network)
* `DCUNet`: Deep Complexed Unet
* `DCCRN`: Deep Complexed Convolutional Recurrent Network
* `DenseUnet`: Unet boosted using DenseBlocks
* `Phasen`: Phasen network
* `CRNet`: Convolutional Recurrent Network for speech enhancement
* `TimeDPRNN`: Time domain DPRNN
* `FreqDPRNN`: Frequency domain DPRNN
* `ToyRNN`: Basic RNN model

Each network should have function `infer()` for inference while `forward()` is left for training only.
