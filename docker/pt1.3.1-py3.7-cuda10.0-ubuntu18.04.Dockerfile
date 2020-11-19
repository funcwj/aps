FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

LABEL description="Dockerfile for APS toolkit"
LABEL tag="pt1.3.1-py3.7-cuda10.0-hvd-ubuntu18.04"
LABEL creator="funcwj"
LABEL time="2020/09/04"

ENV CUDA_VERSION=10.0
ENV PYTHON_VERSION=3.7.7
ENV PYTORCH_VERSION=1.3.1
# https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
ENV OPENMPI=v4.0 OPENMPI_VERSION=4.0.5

# install tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    --allow-downgrades --allow-change-held-packages \
    vim git curl locales htop wget zip unzip \
    sox libsndfile1 cmake g++ gcc make ca-certificates libnccl2 libnccl-dev && \
    locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

# install openmpi (for horovod)
RUN mkdir openmpi && cd openmpi && \
    wget https://download.open-mpi.org/release/open-mpi/${OPENMPI}/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && make install && ldconfig && rm -rf openmpi

ENV PATH=/usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

# install miniconda
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

ENV PATH=/usr/local/cuda/bin:/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:$LD_LIBRARY_PATH

# install python dependency
RUN conda install -y python=$PYTHON_VERSION
RUN conda install -y pytorch==$PYTORCH_VERSION torchvision cudatoolkit=$CUDA_VERSION -c pytorch
RUN conda install -y pyyaml matplotlib tqdm scipy h5py pybind11 yapf
RUN conda install -y -c conda-forge tensorboard pysoundfile librosa pre-commit && conda clean -ya
RUN pip install torch_complex kaldi_python_io editdistance museval pystoi pypesq pytest flake8
RUN pip install warp_rnnt && python -m warp_rnnt.test
RUN pip install https://github.com/kpu/kenlm/archive/master.zip

# install hovorod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig && horovodrun --check-build
