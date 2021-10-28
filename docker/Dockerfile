ARG CUDA=10.2
ARG CUDNN=7

# see https://gitlab.com/nvidia/container-images/cuda/blob/master/doc/supported-tags.md
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.10
ARG OPENMPI=v4.0
ARG OPENMPI_VERSION=4.0.5

LABEL description="Dockerfile for APS toolkit"
LABEL tag="pt${PYTORCH_VERSION}-py${PYTHON_VERSION}-cuda${CUDA}-ubuntu18.04"
LABEL creator="funcwj"
LABEL time="2021/04/08"

# install tools
RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades \
    --allow-change-held-packages vim git curl locales htop wget zip unzip sox libsndfile1 \
    cmake g++ gcc make openssh-server ca-certificates libnccl2 libnccl-dev && \
    locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

# https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
# install openmpi (for horovod, optional)
RUN wget https://download.open-mpi.org/release/open-mpi/${OPENMPI}/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar -zxf openmpi-${OPENMPI_VERSION}.tar.gz && rm openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && ./configure --enable-orterun-prefix-by-default && make -j $(nproc) all && \
    make install && ldconfig && cd ../ && rm -rf openmpi-${OPENMPI_VERSION}

ENV PATH=/usr/local/mpi/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH

# install miniconda
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/local/lib:$LD_LIBRARY_PATH

# install python dependency
RUN conda install -y python=${PYTHON_VERSION}
# see https://pytorch.org/get-started/previous-versions/
RUN conda install -y pytorch==${PYTORCH_VERSION} torchaudio cudatoolkit=${CUDA_TOOLKIT} -c pytorch
RUN conda install -y pyyaml matplotlib tqdm scipy h5py pybind11 yapf
RUN conda install -y -c conda-forge tensorboard pysoundfile librosa pre-commit && conda clean -ya
RUN pip install kaldi_python_io edit-distance museval sentencepiece pystoi pypesq pytest flake8
RUN pip install https://github.com/kpu/kenlm/archive/master.zip
# patch
RUN pip install numba==0.52

# install hovorod (optional)
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig && horovodrun --check-build

# one RNNT implementation (or pip install warp_rnnt)
ENV CFLAGS="-I $CUDA_HOME/include $CFLAGS"
# for CUDA11
RUN git clone https://github.com/ncilfone/warp-transducer.git && cd warp-transducer && \
    mkdir build && cd build && cmake .. && make -j $(nproc) && make install && ldconfig -v && \
    cd ../pytorch_binding && python setup.py install && cd ../../ && rm -rf warp-transducer

# install apex (may abort, optional)
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -rf apex

# install sentencepiece
RUN git clone https://github.com/google/sentencepiece.git && cd sentencepiece && mkdir build && cd build && \
    cmake .. && make -j $(nproc) && make install && ldconfig -v && cd ../../ && rm -rf sentencepiece

# test
# RUN python -m warp_rnnt.test
RUN python -c "from warprnnt_pytorch import rnnt_loss"
