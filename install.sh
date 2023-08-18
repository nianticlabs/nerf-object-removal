#!/bin/bash

# echo "Installing LaMa"

cd external/lama
pip install -r requirements.txt --no-cache
cd ../..

# downloading pretraind lama file
cd external/lama
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
unzip big-lama.zip
cd ../..

echo "Installing jax"
pip install -r requirements_v2.txt --no-cache
pip install pyyaml matplotlib scikit-image
pip install jax==0.3.20 jaxlib==0.3.15+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install flax==0.5.1 chex==0.1.5 optax==0.1.5 oryx==0.2.4 orbax==0.1.7 --no-deps

pip install tensorflow
pip install numpy==1.23.1 # this is important that lama works

