FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget mc tmux nano build-essential rsync libgl1 \
    python3-pip \
    python \
    ffmpeg libsm6 libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt install python-is-python3 -y && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install numpy scipy opencv-python tensorflow joblib matplotlib pandas \
    albumentations==0.5.2 pytorch-lightning==1.2.9 tabulate easydict==1.9.0 kornia==0.5.0 webdataset \
    packaging gpustat tqdm pyyaml hydra-core==1.1.0.dev6 scikit-learn==0.24.2 tabulate
RUN pip install scikit-image==0.17.2

COPY . /app/lama
WORKDIR /app/lama

ENV PYTHONPATH=/app/lama
ENV TORCH_HOME=/app/lama

RUN ln -s /usr/local/cuda /usr/local/cuda-11.2
RUN XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2

ENV USER=root
