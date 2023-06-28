# NeRF ObjectRemoval from Docker

## 1. Build docker image locally

```shell
cd docker
bash ./build_docker.sh
```

## 2. Run docker container interactively

```shell
cd docker
bash ./run_docker.sh
```

## 3. Run method on a scene

```shell
# activate env
micromamba activate object-removal
# choose scene
export SCENE_NUMBER="001"
# run
ROOT_DIR="$(pwd)/data/nerf-object-removal" \
OUTPUT_DIR="$(pwd)/experiments/real" \
bash ./run_real.sh model/configs/custom/default.gin "${SCENE_NUMBER}"
```