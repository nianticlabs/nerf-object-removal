#!/usr/bin/env bash

# if "data" does not exist
if [ ! -n "$(find ../data -type f -print -quit)" ]; then
  mkdir "../data"
  wget -P "../data" "https://storage.googleapis.com/niantic-lon-static/research/nerf-object-removal/nerf-object-removal.zip"
  unzip "../data/nerf-object-removal.zip" -d "../data"
fi

# run docker image
docker run \
  --gpus all \
  -it \
  -v "$(pwd)/../data:/app/object-removal/data" \
  nerf-object-removal:latest \
  bash