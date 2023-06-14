#!/bin/bash

# echo "Installing LaMa"
cd external/lama
pip install -r requirements.txt --no-cache
cd ../..
# downloading pretraind lama file
# pip install wldhx.yadisk-direct
# curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
# unzip big-lama.zip
# rm big-lama.zip
mkdir -p external/lama/big-lama/models
curl -c -L https://huggingface.co/camenduru/big-lama/resolve/main/big-lama/models/best.ckpt -o external/lama/big-lama/models/best.ckpt
curl -c -L https://huggingface.co/camenduru/big-lama/resolve/main/big-lama/config.yaml -o external/lama/big-lama/config.yaml

echo "Installing jax"
pip install -r requirements_v2.txt --no-cache --no-deps
pip install pyyaml matplotlib scikit-image
pip install jax==0.3.20 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.5.1 chex==0.1.5 optax==0.1.5 oryx==0.2.4 orbax==0.1.7 --no-deps

pip install tensorflow