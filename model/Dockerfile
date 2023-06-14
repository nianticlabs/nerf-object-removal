FROM eu.gcr.io/res-interns/silvan-jax:latest

COPY . /app/object-removal
WORKDIR /app/object-removal

RUN pip install tqdm pandas

ENV XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false
ENV TF_CPP_MIN_LOG_LEVEL=0