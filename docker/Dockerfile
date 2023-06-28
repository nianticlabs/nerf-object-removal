# Bring in the micromamba image so we can copy files from it
FROM mambaorg/micromamba:1.4.4 as micromamba

# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# need devel for /usr/local/cuda/bin/ptxas

#
# Install basic dependencies in Ubuntu
#

# Environment variables for non-interactive apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

# Install curl
RUN --mount=type=cache,target=/var/cache/apt \
  apt update && \
  apt install -y curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app


########################################################################
# 
# Install micromamba from official image
#

# Source:
# https://github.com/mamba-org/micromamba-docker#adding-micromamba-to-an-existing-docker-image

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER="root"
ARG MAMBA_USER_ID=0
ARG MAMBA_USER_GID=0
ENV MAMBA_USER="${MAMBA_USER}"
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER "${MAMBA_USER}"

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

#
# END micromamba official image
#
########################################################################

#
# Create conda env
#

RUN micromamba create -n object-removal python=3.8 -c conda-forge

#
# Prepare env variables for installing JAX
#

# LD_LIBRARY_PATH
# libcuda.so.1 is in /usr/local/cuda/compat
# libnvinfer.so is in .../lib/python3.8/site-packages/tensorrt
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/compat:/opt/conda/envs/object-removal/lib/python3.8/site-packages/tensorrt_libs/:/opt/conda/envs/object-removal/lib/python3.8/site-packages/nvidia/cudnn/lib/"

ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
ENV TF_CPP_MIN_LOG_LEVEL=0

#
# Install JAX
#

# jax needs '-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'
# tensorrt_libs needs '--extra-index-url https://pypi.nvidia.com'

RUN --mount=type=cache,target=/root/.cache \
    eval "$(micromamba shell hook --shell=bash)" && \
    micromamba activate object-removal && \
    pip install \
      "jax[cuda11_cudnn82]==0.3.21" \
      "jaxlib==0.3.20+cuda11.cudnn82" \
      "flax==0.5.1" \
      "chex==0.1.5" \
      "optax==0.1.5" \
      "oryx==0.2.4" \
      "tensorflow==2.11.0" \
      "tqdm==4.65" \
      "pandas==2.0.2" \
      "pyyaml==6.0" \
      "matplotlib==3.7.1" \
      "scikit-image==0.17.2" \
      "open3d==0.13.0" \
      "gin-config==0.5" \
      "dm-pix==0.2.1" \
      "numpy==1.23.3" \
      "torchvision==0.15.1" \
      "nvidia-tensorrt==99" \
      "tensorrt_libs==8.6.1" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
      --extra-index-url https://pypi.nvidia.com

# Fix tensorrt version mismatch
RUN ln -s /opt/conda/envs/object-removal/lib/python3.8/site-packages/tensorrt_libs/libnvinfer.so.{8,7} && \
    ln -s /opt/conda/envs/object-removal/lib/python3.8/site-packages/tensorrt_libs/libnvinfer_plugin.so.{8,7}

# Install orbax without upgrading to jax 0.4.2
RUN --mount=type=cache,target=/root/.cache \
    eval "$(micromamba shell hook --shell=bash)" && \
    micromamba activate object-removal && \
    pip install "orbax==0.1.7" --no-deps

#
# Install Lama
#

COPY ./external/lama /app/object-removal/external/lama
RUN --mount=type=cache,target=/root/.cache \
    eval "$(micromamba shell hook --shell=bash)" && \
    micromamba activate object-removal && \
    cd /app/object-removal/external/lama && \
    pip install -r requirements.txt

#
# Copy rest of the source code
#

COPY . /app/object-removal
WORKDIR /app/object-removal