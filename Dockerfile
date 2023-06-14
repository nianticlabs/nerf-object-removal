FROM eu.gcr.io/res-interns/silvan-jax:latest

COPY . /app/object-removal
WORKDIR /app/object-removal


RUN bash install.sh
RUN pip install tqdm pandas

RUN pip install tensorflow==2.11

ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
ENV TF_CPP_MIN_LOG_LEVEL=0