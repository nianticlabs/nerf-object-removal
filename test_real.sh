#!/bin/bash

if [ -z "${ROOT_DIR}" ]; then
    echo "Error, need \${ROOT_DIR} to be set!"
    echo "E.g. ROOT_DIR=\"$(pwd)/data/object-removal-custom\""
    exit 1
fi

if [ -z "${OUTPUT_DIR}" ]; then
    echo "Error, need \${OUTPUT_DIR} to be set!"
    echo "E.g. OUTPUT_DIR=\"$(pwd)/experiments/object-removal-custom/real\""
    exit 1
fi

sed -i.bak \
  -E "s@Config.data_dir = \".*@Config.data_dir = \"${ROOT_DIR}\"@1" \
  model/configs/custom/default.gin
sed -i.bak \
  -E "s@Config.checkpoint_dir = \".*@Config.checkpoint_dir = \"${OUTPUT_DIR}/default\"@1" \
  model/configs/custom/default.gin

CONFIG="$1"
SCENE="$2"
NAME="$3"
# ROOT_DIR=/mnt/res_nas/silvanweder/datasets/object-removal-custom-clean
# OUTPUT_DIR=/mnt/res_nas/silvanweder/experiments/final_tests_real
HOME_DIR=$(pwd)

# make sure this works with an absolute path
# OUTPUT_DIR=/home/ext_silvan_weder_gmail_com/experiments

# setup experiment folder and symlink raw data
mkdir -p ${OUTPUT_DIR}/$SCENE/data
cd ${OUTPUT_DIR}/$SCENE/data
ln -s ${ROOT_DIR}/$SCENE/* .
cd ${HOME_DIR}


# run nerf optimization
export CUDA_VISIBLE_DEVICES=0; 
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python; 

python model/eval_train.py \
        --gin_configs $CONFIG \
        --scene $SCENE \
        --data_dir ${OUTPUT_DIR}/$SCENE/data \
        --checkpoint_dir ${OUTPUT_DIR}/$SCENE

python model/eval.py \
        --gin_configs $CONFIG \
        --scene $SCENE \
        --data_dir ${OUTPUT_DIR}/$SCENE/data \
        --checkpoint_dir ${OUTPUT_DIR}/$SCENE

# remove raw data symlink
cd ${OUTPUT_DIR}/$SCENE/data
# rm *
find "${OUTPUT_DIR}/${SCENE}/data" -type l -delete