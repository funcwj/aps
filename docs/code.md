# Code Document

Providing a simple description of the code structure and design here.

## `aps.transform`

Customized feature transform module in aps. The feature transform layer from `aps.transform` is stacked as the feature extractor for ASR or enhancement/separation tasks, e.g., a stack of `SpectrogramTransform-AbsTransform-MelTransform-LogTransform` can used to extract log mel filter-bank features. Currently the supported feature transform layers are shown in the following:

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

Trainer module in aps. We use `horovod` & PyTorch's & Apex's `DistributedDataParallel` for distributed training. Now the three Trainer instances are provided, which are inherited from the base class in `aps.trainer.Trainer`:

* `HvdTrainer`: Multi-GPU training using horovod
* `DdpTrainer`: Single-GPU or multi-GPU training using PyTorch's DistributedDataParallel
* `ApexTrainer`: Single-GPU or multi-GPU training using Apex's DistributedDataParallel, also aims to mixed precision training

The scheduler of the learning rate and schedule sampling is defined in `aps.trainer.lr` and `aps.trainer.ss`.

## `aps.task`

Supported task in aps. The `Task` class is responsible for the computation of an user-specific objective function, which is defined in the `forward()` function. Actually it's still a PyTorch's `Module` class and each one should inherit from `th.nn.Module`. Currently the supported task are shown below:

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
* `ComplexMappingTask`: frequency domain complex mapping objective function

## `aps.loader`

Supported data loader in aps. For acoustic model training, we have two options

* `am_raw`: Raw waveform data loader which do not need us to prepare acoustic features beforehead (recommended).
* `am_kaldi`: Data loader that supports feature format in Kaldi toolkit.

For enhancement/separation model, we also have two options

* `ss_chunk`: Raw waveform data loader and also no need to prepare features.
* `ss_online`: A data loader in an online manner which generates training audio (noisy, single/multi-speaker, close-talk/far-field) on-the-fly.

## `aps.distributed`

A package to handle distributed training and provide an unified interface. Now we only have two options: `torch` and `horovod`.

## `aps.asr`

The submodule for language model & acoustic model. Currently the implemented AM are:

* `AttASR`: Attention based encoder-decoder AM with RNN based decoder
* `EnhAttASR`: `AttASR` with multi-channel front-end
* `TransformerASR`: Attention based encoder-decoder AM with tran sformer as decoder
* `EnhTransformerASR`: `TransformerASR` with multi-channel front-end
* `TorchTransducerASR`: RNNT AM with RNN based decoder
* `TransformerTransducerASR`: RNNT AM with Transformer based decoder

The transformer implementation is kept similar style with `torch.nn` package and each network should have function `beam_search()` for decoding. Variants of encoder are provided in `aps.asr.base.encoder`:

* `VanillaRNNEncoder`: stack of vanilla RNNs as encoder
* `VariantRNNEncoder`: RNNs with customized features
* `FSMNEncoder`: stack of FSMN as encoder
* `Conv1dEncoder`: stack of TDNN (conv1d) layers as encoder (can also be used for subsampling, following with other encoders).
* `Conv2dEncoder`: stack of conv2d layers (used for subsampling, following with other encoders)

and attention type:
* `DotAttention`: dot attention and its multi-head version `MHDotAttention`
* `LocAttention`: location aware attention and its multi-head version `MHLocAttention`
* `CtxAttention`: context attention and its multi-head version `MHCtxAttention`

## `aps.sse`

The submodule for speech enhancement/separation model. The implemented model are shown below:

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
