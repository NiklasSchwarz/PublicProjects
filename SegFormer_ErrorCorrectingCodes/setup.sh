#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="experiments-env"
PYTHON_VERSION="3.7"
GPU="true" # Set to "false" for CPU-only   

echo ">>> Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

source $(conda info --base)/etc/profile.d/conda.sh
conda activate experiments-env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo ">>> Activated environment: $ENV_NAME"

conda install -c conda-forge libstdcxx-ng

echo ">>> Installing scipy for hadamard matrix with conda"
conda install scipy 

echo ">>> Installing PyTorch with pip (CUDA setting: $GPU)"
python3 -m pip install --upgrade pip setuptools wheel
if [[ "$GPU" == "true" ]]; then
  pip3 install torch torchvision torchaudio 
else
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 
fi

echo ">>> Verifying PyTorch"
python - <<'PY'
import torch, sys, platform
print("PyTorch:", torch.__version__, "| Python:", platform.python_version())
print("CUDA available:", torch.cuda.is_available())
PY

echo ">>> Installing OpenMIM + OpenMMLab packages"
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.0.1"

echo ">>> Installing MMSegmentation and extra libs"
pip3 install "mmsegmentation==1.0.0"
pip3 install reedmuller

echo "Environment $ENV_NAME is ready."
