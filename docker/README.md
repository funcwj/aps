To build docker image:

```shell
# pytorch 1.8.0 with cuda 11.0, python 3.8.5 (now apex failed)
cat Dockerfile | docker build --build-arg CUDA=11.0 \
    --build-arg CUDNN=8 \
    --build-arg CUDA_TOOLKIT=11.1 \
    --build-arg PYTHON_VERSION=3.8.5 \
    --build-arg PYTORCH_VERSION=1.8.0 \
    --tag aps-pt1.8.0-py3.8.5-cuda11.0-ubuntu18.04:v1.0 -
```
