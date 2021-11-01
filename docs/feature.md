# Feature

1. End-to-end training and evaluation. The APS provides PyTorch-based feature extraction for speech tasks (e.g., ASR, SE) and given data and experimental configurations, users can kick off training or evaluation easily.
2. The unified framework for several speech tasks (ASR/SE/SS/SPK). APS aims to reduce the workload for researchers when they run new tasks or train new models. Refer [Q & A](qa.md) to see how to extend your work under the APS framework.
3. Faster ASR decoders. The decoding speed after the optimization (vectorized beam search for attention-based AM) is much faster than the non-parallel version, e.g. used in espnet.
4. Build-in models & objective functions. The APS provides kinds of build-in models & tasks which can be reused directly, e.g., Phasen, DCUNet for SE, DenseUnet, DPRNN for SS, kinds of encoder/decoder variants for E2E-ASR and time/frequency domain objective functions for front-end model training.
5. Less dependency. To use the main feature of the APS, users only need to install several Python packages (some packages shows in `requirements.txt` and `Dockerfile` are optional).
6. Production & Research oriented. The author is going to make APS friendly for both production and research usage.
