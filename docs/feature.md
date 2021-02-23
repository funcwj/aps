# Feature

1. End-to-end training and evaluation. The APS provides PyTorch-based feature extraction for speech tasks (e.g., ASR, SE) and given training data and experimental configurations, the user can kick off training soon.
2. The unified framework for different tasks (ASR/SE/SS). The APS aims to reduce the cost when running new tasks and training new models. Refer [Q & A](qa.md) to see how to add new tasks or models.
3. Optimized module for each task. The code of several modules (e.g., `aps.loader`, `aps.task`, `aps.trainer`) are optimized for several times which can perform with higher efficiency than the early version.
4. Vectorized beam search for attention-based AM. The speed after the optimization is much faster than the non-parallel version, e.g. used in espnet.
4. Build-in models & objective functions. The APS provides kinds of build-in models & task which can be reuse directly, e.g., Phasen, DCUNet for SE, DenseUnet, DPRNN for SS, several encoder/decoder variants for E2E-ASR and time/frequency domain objective functions for front-end model training.
5. Less dependency. There is no much dependency needed to be installed (w/o dependency on Kaldi, some packages shows in `requirements.txt` and `Dockerfile` are optional). But it supports the Kaldi format feature for ASR task as many people use it before.
