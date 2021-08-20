## Overview of the Monthly Update

### 2021/08

1. Kick off streaming ASR and real-time SSE
2. Add streaming features: (i)STFT, fsmn|rnn|conv1d|conv2d ASR encoders
3. Add encode/decode function in EnhTransform class
4. Done with streaming transformer/conformer

### 2021/07

1. Update implementation of DPRNN
2. Add SepFormer & DFSMN & json based online dataloader (for SSE tasks)
3. Begin to add torchscript export testing (dccrn, dcunet, dfsmn, tcn, transformer ...)
4. Change the output of the STFT transform

### 2021/06

1. Setup WHAM recipe
2. Add support for training across the node (multi-nodes)
3. Add DEMUCS model

### 2021/05

1. Setup aishell_v2 & DNS recipe
2. Fix bugs with PyTorch 1.8

### 2021/04

1. Fix bugs in CTC beam search
2. Remove duplicated layernorm in Conformer
3. Update result to WSJ recipe

### 2021/03

1. Unify distributed/non-distributed training command
2. Add stitcher for chunk-wise evaluation in SE/SS task
3. Add CTC alignment command
4. Unify ASR base class (for CTC & Attention & RNNT AM)

### 2021/02

1. Make the repository public
2. Intergrate CTC score & Enable end detection during beam search
3. CTC beam & greedy search
4. Rewrite speed perturb transform

### 2020/01

1. Gradient accumulation & checkpoint average in trainer
2. Rewrite delta feature transform
3. Ngram LM shallow fusion
4. BPTT dataloader for LM
5. Tune and rewrite the beam search decoding
6. Make fbank & mfcc compatible with kaldi
7. Add reduction option to configure asr objective computation
8. Tune and refactor code of transformer/conformer

### 2020/12

1. Stop detector in trainer
2. Add beam search module for ASR
3. Change the backend to for WER computation
4. Add speed perturb transform layer
5. Rewrite beam search for transformer & transducer
6. Add jit LSTM implementation
7. Refactor the transformer codes
8. Add conformer code

### 2020/11

1. Add torchscript examples
2. Use dynamic import & Python decorator
3. Preemphasis in STFT
4. Apex trainer
5. Add flake8 & shellcheck in github workflow
6. Update documents
7. Refactor ASR's encoder code
8. Optimize LM dataloader

### 2020/10

1. Add command to evaluation SE/SS performance
2. Pre-norm & Post-norm & Relative position encoding for Transformer
3. Refactor code of separation task (make it clear and simple)
4. Refactor implementation of Transformer
5. Using python hints

### 2020/09

1. Add docker file to setup environment
2. Setup github workflow
3. Test cases for ASR networks & tasks
4. Fix shallow fusion
5. Add librispeech recipe
6. Make STFT compatiable with librosa (using stft_mode)

### 2020/08

1. Setup CI
2. Add aishell_v1, wsj0_2mix, chime4, timit recipes
3. Test cases for dataloader, task module
4. Add DenseUnet
5. Fix network implementation for SE/SS tasks

### 2020/07

1. Add DCCRN
2. Distributed training for SE/SS task
3. Move learning rate scheduler to a single file
4. Change to absolute import
5. Refactor trainer package (support both horovod and PyTorch's DDP)
6. Test cases for transform module
7. Dataloader for on-the-fly data simulation

### 2020/06

1. Inference command for SE/SS
2. Add WA loss in task package
3. Network implementation update for SE/SS task (add CRN)
4. Document draft

### 2020/05

Refactor the structure to merge the repository that works on speech enhancement/separation

1. Create task module for different tasks (AM, LM, SE & SS)
2. Add documents
3. Add time/frequency loss function for SE/SS task
4. DCUnet & DPRNN & TasNet ...

### 2020/04

1. MVDR front-end with Transformer AM
2. Fix LM dataloader & Add Transformer LM
3. Fix transducer training & decoding
4. Update STFT implementation
5. Add shallow fusion

### 2020/03

1. Multi-channel front-end: fixed beamformer variants
2. Initial code of RNN transducer
3. Unsupervised training experiments on CHiME4
4. Refactor code of Transformer and fix bugs
5. FSMN encoder

### 2020/02

1. Add dataloader for enhancement/separation task
2. SpecAugment

### 2019/12

1. Early stop in Trainer
2. Joint attention & CTC training
3. Global cmvn in feature transform
4. Support distributed training
5. Multi-channel front-end: fixed beamformer (google & amazon)

### 2019/11

1. Initial code of Transformer
2. Vectorized & Batch beam search for decoding
3. Support mfcc extraction & cmvn & feature splicing in the feature transform
4. TDNN encoder
5. RNN LM training & Ngram rescore
6. Schedule sampling during training
7. Multi-head attention variants
8. Multi-channel front-end: MVDR beamformer
9. WER evaluation

### 2019/10

1. Raw audio dataloader
2. Create asr transform module to handle feature extraction
3. Update the Trainer

### 2019/05 - 2019/09

Work on another speech enhancement & separation repository

1. Single & Multi-channel feature extraction (IPD, DF, ...)
2. Time & Frequency domain objective function
3. Network variants in speech enhancement/separation
4. Training and inference commands
5. Dataloader for the offline data pairs and on-the-fly data simulation

### 2019/06

Create the repository and make the first commit to synchronize local data

### 2019/04

Local coding on LAS (listen, attend and spell) experiments

1. First version of the feature transform for ASR task
2. Kaldi feature dataloader
3. RNN encoder & decoder
4. Attention variants (context & dot & location aware)
5. Training & Decoding command
