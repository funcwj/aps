# APS: ASR + Speech Front-End

My own repository for single/multi-channel speech enhancement & speech separation & E2E speech recognition task.

## Support

1. Single channel & Multi-channel acoustic model (Transducer & Encoder-Decoder structure with RNN, Transformer, TDNN, FSMN, ...)
2. Freqency & Time domain single & multi-channel speech enhancement/separation model
3. Distributed training (now using Pytorch's DistributedDataParallel)
4. Kaldi features & PyTorch-based feature extraction

## Quick Start

* [Speech Recognition](doc/recognition.md)
* [Speech Enhancement](doc/enhancement.md)
* [Speech Separation](doc/separation.md)