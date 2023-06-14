# ObjectRemoval

An Nvidia GPU is required to run this code. It's been tested on Ubuntu 20.04.4 LTS.

## Installation

First, you need to clone this repo:

```shell
git clone --recursive git@github.com:nianticlabs/nerf-object-removal.git
```

In order to install this repository, run the following commands. You might need to adjust the CUDA and cudNN versions in [install.sh](install.sh).

```shell
conda create -n object-removal python=3.8
conda activate object-removal
# make sure libcudart.so can be found:
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.8/site-packages/nvidia/cuda_runtime/lib/:${LD_LIBRARY_PATH}"
bash ./install.sh
```

## Download Dataset

Please the download the data from here using the following command:

```shell
mkdir "$(pwd)/data"
cd "$(pwd)/data"
wget -P "$(pwd)/data" "https://storage.googleapis.com/niantic-lon-static/research/nerf-object-removal/nerf-object-removal.zip"
unzip "$(pwd)/data/nerf-object-removal.zip" -d "$(pwd)/data"
```

## Run End-to-end Optimization

In order to test our pipeline using the default configuration on a scene of the provided dataset, run the following command:

```shell
export SCENE_NUMBER="001"; \
ROOT_DIR="$(pwd)/data/object-removal-custom-clean" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./run_real.sh model/configs/custom/default.gin "${SCENE_NUMBER}"
```

or on the synthetic dataset with

```shell
export SCENE_NUMBER="001"; \
ROOT_DIR="$(pwd)/data/object-removal-custom-clean" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./run_synthetic.sh model/configs/custom_synthetic/default.gin "${SCENE_NUMBER}"
```

You have to adjust the ```$EXPERIMENT_PATH``` and the ```$DATA_PATH``` accordingly in the files above.


## Test a trained model

You can test a trained model using the following command for a real scene

```shell
export SCENE_NUMBER="001"; \
ROOT_DIR="$(pwd)/data/object-removal-custom-clean" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./test_real.sh model/configs/custom/default.gin "${SCENE_NUMBER}"
```

or the following command for a synthetic scene

```shell
export SCENE_NUMBER="001"; \
ROOT_DIR="$(pwd)/data/object-removal-custom-clean" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./test_synthetic.sh model/configs/custom_synthetic/default.gin "${SCENE_NUMBER}"
```


## Visualizing the results

Once the models have been optimized, you can visualize the results using [this notebook](notebooks/vis_experiment.ipynb).

Make sure that you correctly set the ```$EXPERIMENT_DIR``` and the ```$GROUNDTRUTH_DIR``` and follow the notebook carefully to set all required options for your experiment. 

## Evaluating the results

You can evaluate the results using the [evaluation script](evaluation/eval.py).

```shell
python eval.py --experiment "${EXPERIMENT_NAME}" --experiment_root_dir "${EXPERIMENT_DIR}" --benchmark (synthetic/real)
```

## Visualizing the evaluation results

You van visualize the evaluation results using [this notebook](notebooks/vis_results.ipynb).

## Citation

If you find this work helpful, please consider citing

```
@article{DBLP:journals/corr/abs-2212-11966,
  author       = {Silvan Weder and
                  Guillermo Garcia{-}Hernando and
                  {\'{A}}ron Monszpart and
                  Marc Pollefeys and
                  Gabriel J. Brostow and
                  Michael Firman and
                  Sara Vicente},
  title        = {Removing Objects From Neural Radiance Fields},
}
```
