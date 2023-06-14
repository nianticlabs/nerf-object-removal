#!/bin/bash

DATA_DIR=$1
OUTPUT_DIR=$2
SCENE=$3
SUFFIX=$4

# setup lama env
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

python prepare_images_custom.py --root_dir $DATA_DIR --output_dir $OUTPUT_DIR --scene $SCENE --suffix $SUFFIX
python prepare_depth_custom.py --root_dir $DATA_DIR --output_dir $OUTPUT_DIR --scene $SCENE --suffix $SUFFIX

python run_depth_custom.py --rotate --root_dir $OUTPUT_DIR --scene $SCENE --suffix $SUFFIX
python run_images_custom.py --rotate --resize --root_dir $OUTPUT_DIR --scene $SCENE --suffix $SUFFIX