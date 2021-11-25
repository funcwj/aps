# Notes

A old example showcasing the way to perform speech enhancement (single-channel) using TorchScript. The pre-trained model is based on [Deep Complex U-Net](https://openreview.net/pdf?id=SkeRTsAcYm) structure.

## Compile & Test
```bash
# depend on sox for audio IO
export SOX_ROOT=/path/to/sox/root
mkdir build & cd build
# compile
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
# test
./build/dcunet-enhan asset/traced_unet.pt data/noisy.wav data/enhan.wav
```
