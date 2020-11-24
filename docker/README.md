To build docker image:

```shell
# pytorch 1.3.1 with cuda 10.0, python 3.7.7
docker build --build-arg CUDA_VERSION=10.0 \
    --build-arg CUDNN=7 \
    --build-arg PYTHON_VERSION=3.7.7 \
    --build-arg PYTORCH_VERSION=1.3.1 \
    --tag aps-pt1.3.1-py3.7.7-cuda10.0-hvd-ubuntu18.04:v1.0 .
# pytorch 1.6.0 with cuda 10.2, python 3.8
docker build --build-arg CUDA_VERSION=10.2 \
    --build-arg CUDNN=7 \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.6.0 \
    --tag aps-pt1.6.0-py3.8.0-cuda10.2-hvd-ubuntu18.04:v1.0 .
# pytorch 1.7.0 with cuda 11.0, python 3.8
docker build --build-arg CUDA_VERSION=11.0 \
    --build-arg CUDNN=7 \
    --build-arg PYTHON_VERSION=3.8 \
    --build-arg PYTORCH_VERSION=1.7.0 \
    --tag aps-pt1.7.0-py3.8.0-cuda11.0-hvd-ubuntu18.04:v1.0 .
```
