# Feature

1. End-to-end training and evaluation. The APS provides PyTorch-based feature extraction for speech tasks (e.g., ASR, SE) and given training data pairs and experimental configurations, the user can kick off training soon.

2. The unified framework for different tasks (ASR/SE/SS). The APS aims to reduce the cost when running new tasks and training new models. Refer [Q & A](qa.md) to see how to add new tasks or models.

3. Optimized module for each task. The code of several modules (e.g., `aps.loader`, `aps.task`) are optimized for several times which performed with high efficiency.

4. Build-in models. The APS provides kinds of build-in models for each task, e.g., Phasen, DCUNet for SE, DenseUnet, DPRNN for SS and several encoder/decoder variants for E2E ASR.

4. Less dependency. There is no much dependency needed to be installed (some packages shows in `requirements.txt` and `Dockerfile` are optional). But it supports the Kaldi format feature for ASR task as many people use it before.
