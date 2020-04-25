# End2End ASR

My repository for E2E acoustic model training.

## Support

1. Single channel AM (Transducer & Encoder-Decoder structure with RNN, Transformer, TDNN, FSMN, ...)
2. Multi-channel AM (Mainly based on popular neural beamforming methods)
3. RNN & Transformer LM
4. Distributed training (now using Pytorch's DistributedDataParallel)
5. Kaldi & Pytorch's feature extraction