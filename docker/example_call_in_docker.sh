#!/usr/bin/env bash

# make sure the environment is activated!
# micromamba activate object-removal

export SCENE_NUMBER="001"
ROOT_DIR="$(pwd)/data/nerf-object-removal" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./run_real.sh model/configs/custom/default.gin "${SCENE_NUMBER}"