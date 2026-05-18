#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cuda-mapdiffusion-temporal}"

echo "Creating conda env: $ENV_NAME"

conda env create -f env/environment.yml -n "$ENV_NAME" || {
  echo "conda env create failed. If the env already exists, run:"
  echo "  conda activate $ENV_NAME"
  exit 1
}

conda activate "$ENV_NAME"

echo "============================================================"
echo "Environment created."
echo "Next steps:"
echo "1. Install PyTorch 1.9.0 + CUDA 11.1 if not already installed."
echo "2. Install MMCV/MMDetection/MMDetection3D matching the original stack."
echo "3. Install or point setup_env.sh to TensorRT 8.5.3.1."
echo "============================================================"
