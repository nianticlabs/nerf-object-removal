#!/usr/bin/env bash

# check if current directory is "docker"
if [ ! -d "external" ]; then
  # go up to project root
  pushd ..
fi

# check if current directory is project root
if [ ! -d "external" ]; then
  echo "Please run this script from project root or docker directory"
  exit 1
fi

# check, if "external/lama/big-lama/models/best.ckpt" exists
if [ ! -f "external/lama/big-lama/models/best.ckpt" ]; then
  echo "Downloading pretrained lama file"
  mkdir -p external/lama/big-lama/models
  curl -L https://huggingface.co/camenduru/big-lama/resolve/main/big-lama/models/best.ckpt -o external/lama/big-lama/models/best.ckpt
  curl -L https://huggingface.co/camenduru/big-lama/resolve/main/big-lama/config.yaml -o external/lama/big-lama/config.yaml
fi

DOCKER_BUILDKIT=1 \
docker builder build \
  --progress=plain \
  -t "nerf-object-removal:latest" \
  -f docker/Dockerfile \
  .

# back to docker
if [ -d "external" ]; then
  popd
fi