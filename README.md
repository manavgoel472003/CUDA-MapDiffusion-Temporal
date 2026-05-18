# CUDA-MapDiffusion-Temporal

TensorRT/CUDA runtime for **Temporal MapDiffusion** HD map vector prediction.

This repository contains the CUDA-BEVFusion-based deployment code for running a Temporal MapDiffusion pipeline with TensorRT engines.

The working runtime path is:

```text
multi-view images
  -> Engine A: camera backbone + FPN
  -> Engine B: BEVFormer encoder
  -> Engine D: StreamFusionNeck / warped temporal BEV fusion
  -> Engine C_first: MapDiffusion head for first frame in a scene
  -> Engine C_temporal: MapDiffusion temporal head for later frames
  -> submission_vector.json
  -> vector-map evaluation
```

The important implementation detail is that the MapDiffusion head must receive the **fused/warped BEV** from the temporal fusion neck, not the raw BEV encoder output.

---

## 1. Large artifacts

TensorRT `.plan` engines and compiled `.so` plugin libraries are not tracked in Git.

Download the artifacts from:

```text
PASTE_ARTIFACT_LINK_HERE
```

After downloading, place them exactly like this:

```text
CUDA-MapDiffusion-Temporal/
  model/mapdiffusion_temporal_routeB/build/
    camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan
    camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan
    stream_fusion_neck.temporal87000.fp32.plan
    mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan
    mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan

  build/
    libmapdiffusion_msda.so
    plugins/
      libmmcv_dcnv2_trt.so
      libbevformer_tsa_trt.so
      libbevformer_sca_trt.so
```

Check:

```bash
ls -lh model/mapdiffusion_temporal_routeB/build/*.plan
ls -lh build/libmapdiffusion_msda.so
ls -lh build/plugins/*.so
```

---

## 2. External dependencies

This repo assumes the same environment used for the original Temporal MapDiffusion deployment.

Expected stack:

```text
Python 3.8
PyTorch 1.9.0 + CUDA 11.1
TensorRT 8.5.3.1
MMCV / MMDetection / MMDetection3D
nuScenes trainval data
MapDiffusion Python plugin code
```

On the original HPC setup:

```bash
conda activate streammapnet
```

The external MapDiffusion repo is expected at:

```text
/home/018198687/Mapping/mapdiffusion
```

If your path is different, edit `MAPDIFF_ROOT` in `setup_env.sh`.

---

## 3. Setup

Create `setup_env.sh` in the repo root:

```bash
cat > setup_env.sh <<'SH'
#!/bin/bash

export CUDA_MAPDIFF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_BEV_ROOT="$CUDA_MAPDIFF_ROOT"

export MAPDIFF_ROOT=/home/018198687/Mapping/mapdiffusion

export TRT_ROOT=/home/018198687/Mapping/local/TensorRT-8.5.3.1
export SMN_TORCH_LIB=/home/018198687/ls/envs/streammapnet/lib/python3.8/site-packages/torch/lib
export CUDA11_RUNTIME=/home/018198687/Mapping/local/cuda11_runtime_libs_from_other_env
export NVHPC_MATH_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib

export LD_LIBRARY_PATH="$TRT_ROOT/lib:$NVHPC_MATH_LIB:$SMN_TORCH_LIB:/usr/local/cuda-11.1/lib64:$CUDA11_RUNTIME:$LD_LIBRARY_PATH"
export PYTHONPATH="$CUDA_MAPDIFF_ROOT:$MAPDIFF_ROOT:$PYTHONPATH"

export CONFIG="$CUDA_MAPDIFF_ROOT/model/mapdiffusion_temporal_routeB/temporal_config.py"
export ENGINE_DIR="$CUDA_MAPDIFF_ROOT/model/mapdiffusion_temporal_routeB/build"
export PLUGIN_DIR="$CUDA_MAPDIFF_ROOT/build/plugins"

echo "CUDA_MAPDIFF_ROOT=$CUDA_MAPDIFF_ROOT"
echo "MAPDIFF_ROOT=$MAPDIFF_ROOT"
echo "CONFIG=$CONFIG"
echo "ENGINE_DIR=$ENGINE_DIR"
echo "PLUGIN_DIR=$PLUGIN_DIR"
SH

chmod +x setup_env.sh
```

Load it:

```bash
conda activate streammapnet
source setup_env.sh
```

Test:

```bash
python - <<'PY'
import os
import tensorrt as trt
import torch

print("TensorRT:", trt.__version__)
print("Torch:", torch.__version__, torch.version.cuda)
print("CONFIG exists:", os.path.exists(os.environ["CONFIG"]))
print("ENGINE_DIR exists:", os.path.exists(os.environ["ENGINE_DIR"]))
print("PLUGIN_DIR exists:", os.path.exists(os.environ["PLUGIN_DIR"]))
PY
```

Expected:

```text
TensorRT: 8.5.3.1
Torch: 1.9.0+cu111
CONFIG exists: True
ENGINE_DIR exists: True
PLUGIN_DIR exists: True
```

---

## 4. Smoke test: 2-sample inference

```bash
conda activate streammapnet
source setup_env.sh

OUT="$CUDA_MAPDIFF_ROOT/runs/smoke_2"
rm -rf "$OUT"
mkdir -p "$OUT"

CUDA_LAUNCH_BLOCKING=1 \
python ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py \
  --config "$CONFIG" \
  --out-dir "$OUT" \
  --start 0 \
  --limit 2 \
  --seed 123 \
  2>&1 | tee "$OUT/run.log"
```

Check:

```bash
grep -n "score_max\|saved submission\|Traceback\|Error" "$OUT/run.log"
ls -lh "$OUT/submission_vector.json"
```

A successful smoke test should show high scores for the first two samples and save:

```text
runs/smoke_2/submission_vector.json
```

In the validated self-contained run, both engine/plugin paths loaded from the repo and the first two samples produced `score_max` around `0.92`.

---

## 5. Full validation inference

Full nuScenes validation has 5981 samples.

```bash
conda activate streammapnet
source setup_env.sh

OUT="$CUDA_MAPDIFF_ROOT/runs/full_val"
rm -rf "$OUT"
mkdir -p "$OUT"

CUDA_LAUNCH_BLOCKING=1 \
python ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py \
  --config "$CONFIG" \
  --out-dir "$OUT" \
  --start 0 \
  --limit 5981 \
  --seed 123 \
  2>&1 | tee "$OUT/run.log"
```

Output:

```text
runs/full_val/submission_vector.json
```

---

## 6. Chunked full validation

```bash
conda activate streammapnet
source setup_env.sh

export BASE_OUT="$CUDA_MAPDIFF_ROOT/runs/full_val_chunks"
export TOTAL=5981
export CHUNK=500
export SEED=123

rm -rf "$BASE_OUT"
mkdir -p "$BASE_OUT"

ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_chunks.sh \
  2>&1 | tee "$BASE_OUT/full_chunk_run.log"
```

Each chunk writes:

```text
runs/full_val_chunks/chunk_*/submission_vector.json
```

---

## 7. Evaluate submission

```bash
conda activate streammapnet
source setup_env.sh

SUB="$CUDA_MAPDIFF_ROOT/runs/full_val/submission_vector.json"
EVAL_OUT="$CUDA_MAPDIFF_ROOT/runs/full_val/eval_direct_metric"
mkdir -p "$EVAL_OUT"

python scripts/eval_submission_direct.py \
  --config "$CONFIG" \
  --submission "$SUB" \
  --out-dir "$EVAL_OUT" \
  2>&1 | tee "$EVAL_OUT/eval.log"
```

Summarize:

```bash
grep -n "category\|ped_crossing\|divider\|boundary\|mAP_normal\|eval_res" \
  "$EVAL_OUT/eval.log"
```

---

## 8. Important runtime notes

### Correct BEV input

The MapDiffusion head must use:

```text
Engine D output: fused_bev
```

Do not feed raw BEV encoder output into the MapDiffusion head for the final temporal pipeline.

### First-frame vs temporal head

Use:

```text
C_first      for the first frame in each scene
C_temporal   for subsequent frames
```

### Temporal state

The runtime keeps and updates:

```text
prev_bev_state
prev_query_feat_state
```

Both reset at scene boundaries.

### Diagnostic flags

Use only for parity debugging:

```text
PT_QUERY_REPLAY=1
PT_PREVQ_REPLAY=1
PT_BEV_REPLAY=1
PT_PARITY_NO_D_NO_QMEM=1
```

---

## 9. Troubleshooting

### `ImportError: libnvinfer.so.8`

Check TensorRT path:

```bash
export TRT_ROOT=/home/018198687/Mapping/local/TensorRT-8.5.3.1
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$LD_LIBRARY_PATH"
```

Test:

```bash
python - <<'PY'
import tensorrt as trt
print(trt.__version__)
PY
```

### `ImportError: libcublas.so.11`

Add CUDA/Torch/NVHPC libraries:

```bash
export SMN_TORCH_LIB=/home/018198687/ls/envs/streammapnet/lib/python3.8/site-packages/torch/lib
export CUDA11_RUNTIME=/home/018198687/Mapping/local/cuda11_runtime_libs_from_other_env
export NVHPC_MATH_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib

export LD_LIBRARY_PATH="$TRT_ROOT/lib:$NVHPC_MATH_LIB:$SMN_TORCH_LIB:/usr/local/cuda-11.1/lib64:$CUDA11_RUNTIME:$LD_LIBRARY_PATH"
```

### Plugin deserialization error

Make sure the plugin libraries exist:

```bash
ls -lh build/libmapdiffusion_msda.so
ls -lh build/plugins/*.so
```

The runner loads these before deserializing TensorRT engines.

---

## 10. Minimal command sequence

```bash
git clone https://github.com/manavgoel472003/CUDA-MapDiffusion-Temporal.git
cd CUDA-MapDiffusion-Temporal

# Download artifacts and place them under:
# model/mapdiffusion_temporal_routeB/build/
# build/
# build/plugins/

conda activate streammapnet
source setup_env.sh

OUT="$CUDA_MAPDIFF_ROOT/runs/smoke_2"
mkdir -p "$OUT"

CUDA_LAUNCH_BLOCKING=1 \
python ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py \
  --config "$CONFIG" \
  --out-dir "$OUT" \
  --start 0 \
  --limit 2 \
  --seed 123 \
  2>&1 | tee "$OUT/run.log"
```
