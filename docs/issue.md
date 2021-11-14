# Known Issues

1. The configuration & recipe examples may be out-of-date so please follow the commit hash of the git for reproducing.
2. The PyTorch-based feature extraction (for ASR tasks) are not guaranted to be same (but similar) as [Kaldi](https://github.com/kaldi-asr/kaldi), but it can be modified easily if you are familiar with the extraction process in Kaldi.
3. The implementation of the APS's network module (i.e., `aps.asr`, `aps.sse`) is based on author's personal knowledge thus the difference between the codes and original papers may exist.
