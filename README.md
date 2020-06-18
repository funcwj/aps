# APS: ASR + Speech Front-End

My repository for single/multi-channel speech enhancement & speech separation & E2E speech recognition task.

## Support

1. Single channel acoustic model (Transducer & Encoder-Decoder structure with RNN, Transformer, TDNN, FSMN, ...)
2. Multi-channel acoustic model (Mainly based on the neural beamforming methods)
3. Freqency & Time domain single & multi-channel speech enhancement/separation models
4. Distributed training (now using Pytorch's DistributedDataParallel)
5. Kaldi features & PyTorch-based feature extraction

## Quick Start

* [Speech Recognition](doc/recognition.md)
* [Speech Enhancement](doc/enhancement.md)
* [Speech Separation](doc/separation.md)