# Conda Environment Setup for CUDA-MapDiffusion-Temporal

This file documents the environment setup for the CUDA-MapDiffusion-Temporal repo.

The original working environment was an HPC conda environment named:

```bash
streammapnet
```

The tested runtime stack was approximately:

```text
Python 3.8
PyTorch 1.9.0 + CUDA 11.1
TorchVision 0.10.0 + CUDA 11.1
MMCV 1.6.0
MMDetection 2.28.2
MMDetection3D 1.0.0rc6
TensorRT 8.5.3.1
```

TensorRT engines were built and tested with TensorRT 8.5.3.1, so use the same TensorRT version when possible.

---

## 1. Add an `env/` folder to the repo

From the repo root:

```bash
cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

mkdir -p env
```

---

## 2. Export the exact working conda environment

Activate the working environment:

```bash
conda activate streammapnet
```

Export the full conda environment:

```bash
conda env export --no-builds > env/environment.full.yml
```

Export the full pip package list:

```bash
pip freeze > env/requirements.full.txt
```

These two files are useful for exact HPC reproduction:

```text
env/environment.full.yml
env/requirements.full.txt
```

---

## 3. Add a cleaner installable `environment.yml`

Create:

```text
env/environment.yml
```

with this content:

```yaml
name: cuda-mapdiffusion-temporal
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.8
  - pip
  - numpy
  - scipy
  - tqdm
  - pyyaml
  - scikit-learn
  - matplotlib
  - shapely
  - opencv
  - pyquaternion
  - pillow
  - protobuf
  - pip:
      - nuscenes-devkit
      - lyft-dataset-sdk
      - networkx
      - descartes
      - einops
      - timm
      - yapf==0.40.1
```

Create it from the shell:

```bash
cat > env/environment.yml <<'YAML'
name: cuda-mapdiffusion-temporal
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.8
  - pip
  - numpy
  - scipy
  - tqdm
  - pyyaml
  - scikit-learn
  - matplotlib
  - shapely
  - opencv
  - pyquaternion
  - pillow
  - protobuf
  - pip:
      - nuscenes-devkit
      - lyft-dataset-sdk
      - networkx
      - descartes
      - einops
      - timm
      - yapf==0.40.1
YAML
```

---

## 4. Add a helper script

Create:

```text
env/create_env.sh
```

with this content:

```bash
#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cuda-mapdiffusion-temporal}"

echo "Creating conda env: $ENV_NAME"

conda env create -f env/environment.yml -n "$ENV_NAME" || {
  echo "conda env create failed. If the env already exists, run:"
  echo "  conda activate $ENV_NAME"
  exit 1
}

echo "============================================================"
echo "Environment created."
echo "Next steps:"
echo "1. conda activate $ENV_NAME"
echo "2. Install PyTorch 1.9.0 + CUDA 11.1 if not already installed."
echo "3. Install MMCV/MMDetection/MMDetection3D matching the original stack."
echo "4. Install or point setup_env.sh to TensorRT 8.5.3.1."
echo "============================================================"
```

Create it from the shell:

```bash
cat > env/create_env.sh <<'SH'
#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cuda-mapdiffusion-temporal}"

echo "Creating conda env: $ENV_NAME"

conda env create -f env/environment.yml -n "$ENV_NAME" || {
  echo "conda env create failed. If the env already exists, run:"
  echo "  conda activate $ENV_NAME"
  exit 1
}

echo "============================================================"
echo "Environment created."
echo "Next steps:"
echo "1. conda activate $ENV_NAME"
echo "2. Install PyTorch 1.9.0 + CUDA 11.1 if not already installed."
echo "3. Install MMCV/MMDetection/MMDetection3D matching the original stack."
echo "4. Install or point setup_env.sh to TensorRT 8.5.3.1."
echo "============================================================"
SH

chmod +x env/create_env.sh
```

Run it with:

```bash
bash env/create_env.sh
```

Or create the environment directly:

```bash
conda env create -f env/environment.yml
conda activate cuda-mapdiffusion-temporal
```

---

## 5. Add OpenMMLab install notes

Create:

```text
env/openmmlab_install_notes.md
```

with this content:

```markdown
# OpenMMLab / CUDA Stack Notes

The original working environment used approximately:

- Python 3.8
- PyTorch 1.9.0 + CUDA 11.1
- TorchVision 0.10.0 + CUDA 11.1
- MMCV 1.6.0
- MMDetection 2.28.2
- MMDetection3D 1.0.0rc6
- TensorRT 8.5.3.1

TensorRT was loaded from:

/home/018198687/Mapping/local/TensorRT-8.5.3.1

The runtime also used additional CUDA/Torch library paths through setup_env.sh.

If installing on a new machine, install the OpenMMLab packages according to the machine's CUDA and PyTorch version.

Important: TensorRT .plan engines are not fully portable across arbitrary TensorRT/GPU/driver combinations. The closest match is TensorRT 8.5.3.1 with the same CUDA runtime stack.
```

Create it from shell:

```bash
cat > env/openmmlab_install_notes.md <<'MD'
# OpenMMLab / CUDA Stack Notes

The original working environment used approximately:

- Python 3.8
- PyTorch 1.9.0 + CUDA 11.1
- TorchVision 0.10.0 + CUDA 11.1
- MMCV 1.6.0
- MMDetection 2.28.2
- MMDetection3D 1.0.0rc6
- TensorRT 8.5.3.1

TensorRT was loaded from:

/home/018198687/Mapping/local/TensorRT-8.5.3.1

The runtime also used additional CUDA/Torch library paths through setup_env.sh.

If installing on a new machine, install the OpenMMLab packages according to the machine's CUDA and PyTorch version.

Important: TensorRT .plan engines are not fully portable across arbitrary TensorRT/GPU/driver combinations. The closest match is TensorRT 8.5.3.1 with the same CUDA runtime stack.
MD
```

---

## 6. TensorRT runtime setup

TensorRT is not fully handled by the conda YAML in the original working setup.

The repo should include a `setup_env.sh` file in the root. It should point to TensorRT 8.5.3.1 and CUDA libraries.

Example:

```bash
export TRT_ROOT=/home/018198687/Mapping/local/TensorRT-8.5.3.1
export SMN_TORCH_LIB=/home/018198687/ls/envs/streammapnet/lib/python3.8/site-packages/torch/lib
export CUDA11_RUNTIME=/home/018198687/Mapping/local/cuda11_runtime_libs_from_other_env
export NVHPC_MATH_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib

export LD_LIBRARY_PATH="$TRT_ROOT/lib:$NVHPC_MATH_LIB:$SMN_TORCH_LIB:/usr/local/cuda-11.1/lib64:$CUDA11_RUNTIME:$LD_LIBRARY_PATH"
```

Test TensorRT:

```bash
source setup_env.sh

python - <<'PY'
import tensorrt as trt
import torch

print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__, torch.version.cuda)
PY
```

Expected:

```text
TensorRT: 8.5.3.1
Torch: 1.9.0+cu111
```

---

## 7. Add this section to the main README

Copy this into `README.md`:

```markdown
## Conda environment setup

This repo includes environment helper files under:

```text
env/
  environment.yml
  environment.full.yml
  requirements.full.txt
  create_env.sh
  openmmlab_install_notes.md
```

The original working environment was:

```text
Python 3.8
PyTorch 1.9.0 + CUDA 11.1
TorchVision 0.10.0 + CUDA 11.1
MMCV 1.6.0
MMDetection 2.28.2
MMDetection3D 1.0.0rc6
TensorRT 8.5.3.1
```

Create a fresh conda environment:

```bash
conda env create -f env/environment.yml
conda activate cuda-mapdiffusion-temporal
```

Or use the helper:

```bash
bash env/create_env.sh
```

For exact HPC reproduction, inspect:

```text
env/environment.full.yml
env/requirements.full.txt
```

TensorRT is not fully handled by the conda YAML in this setup. Point `TRT_ROOT` in `setup_env.sh` to your TensorRT 8.5.3.1 install, then run:

```bash
source setup_env.sh

python - <<'PY'
import tensorrt as trt
import torch
print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__, torch.version.cuda)
PY
```

If this import fails, fix `TRT_ROOT`, `LD_LIBRARY_PATH`, or the CUDA library paths in `setup_env.sh`.
```

---

## 8. Commit environment files

```bash
git add env/environment.yml \
        env/environment.full.yml \
        env/requirements.full.txt \
        env/create_env.sh \
        env/openmmlab_install_notes.md \
        README.md

git status
git commit -m "Add conda environment setup docs"
git push
```

If `environment.full.yml` or `requirements.full.txt` is too noisy, commit only:

```bash
git add env/environment.yml env/create_env.sh env/openmmlab_install_notes.md README.md
git commit -m "Add conda environment setup docs"
git push
```
