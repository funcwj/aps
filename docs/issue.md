# Known issues

1. The recipes under [examples/asr](../examples/asr) are not guaranted to achieve STOA results. Previously the part of the ASR code is wrote for internal dataset.
2. The code in each module are refactored and optimized several times thus the configuration & recipe examples may be out-of-date.
3. At the early stage, the APS aims to support jit exporting features which could simplify the cost of model deployment. Currently not all the supported models are tested for this feature.
4. The PyTorch-based feature extraction (ASR part) are not guaranted to get same (but similar) results as [Kaldi](https://github.com/kaldi-asr/kaldi), but it can be modified easily if you are familiar with the extraction process in Kaldi.
5. The implementation of the APS's network module (i.e., `aps.asr`, `aps.sse`) is based on author's personal knowledge thus the mismatch between the code and paper may exist.
