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

## Key result: inference latency

This deployment ports Temporal MapDiffusion inference from the original PyTorch/ONNX-style runtime into a TensorRT/CUDA pipeline.

| Runtime mode | Approx. latency / frame | Notes |
|---|---:|---|
| Before optimization | ~300 ms | Original non-optimized model/runtime path. |
| TensorRT FP32 | ~70 ms | Optimized CUDA/TensorRT runtime with FP32 engines. |
| TensorRT FP16 | TBD / add measured number | Add the final measured FP16 latency once benchmarked on the target GPU. |

Approximate confirmed speedup from the optimized FP32 TensorRT path:

```text
300 ms -> 70 ms
~4.3x faster
```

Latency depends on GPU, TensorRT build flags, precision, batch size, I/O overhead, and whether timing includes preprocessing/postprocessing.

---

## Repo structure

Expected repository layout:

```text
CUDA-MapDiffusion-Temporal/
├── README.md
├── setup_env.sh                         # local environment setup
├── src/                                 # CUDA/C++ source and TensorRT plugin source
│   ├── common/
│   ├── bevfusion/
│   ├── onnx/
│   └── plugins/
├── tool/                                # CUDA-BEVFusion build/helper scripts
├── tools/
│   └── temporal_routeB/                 # ONNX export scripts for Temporal Route-B
├── ports/
│   └── mapdiffusion_temporal_routeB/
│       ├── export/                      # export/build helper scripts
│       ├── paths/                       # path helpers
│       ├── parity/                      # parity/debug scripts
│       └── run/
│           ├── run_temporal_routeB_val_submission.py
│           ├── run_temporal_routeB_chunks.sh
│           ├── compare_e2e_traces.py
│           └── visualization / replay utilities
├── model/
│   └── mapdiffusion_temporal_routeB/
│       ├── temporal_config.py
│       └── build/                       # downloaded TensorRT .plan files go here
├── build/
│   ├── libmapdiffusion_msda.so           # downloaded plugin library
│   └── plugins/                          # downloaded plugin libraries
├── scripts/
│   └── eval_submission_direct.py         # direct metric eval from submission_vector.json
├── artifacts/
│   └── videos/
│       └── output_demo.mp4               # optional manually uploaded output video
└── runs/                                # generated inference/eval outputs, not committed
```

Folders that are generated during inference/eval and should usually stay out of Git:

```text
runs/
debug*/
work_dirs/
eval_work_dir/
*_trace*/
*_debug*/
trt_eval*/
e2e_trace*/
```

---

## Exact engine/plugin artifacts to download

TensorRT `.plan` engines and compiled `.so` plugin libraries are not tracked in Git.

Download the artifacts from:

```text
https://drive.google.com/drive/folders/15zWdbM1xNcBinowPqMhtT0hnobUEdcGr?usp=drive_link
```

After downloading, place the files exactly as shown below.

### Required TensorRT plan files

Put these **five `.plan` files** in:

```text
model/mapdiffusion_temporal_routeB/build/
```

Required files:

```text
model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan
model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan
model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan
model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan
model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan
```

### Optional build logs

These are optional and only useful for reproducibility/debugging. They are not required for inference:

```text
model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.build.log
model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.build.log
model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.build.log
model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.build.log
model/mapdiffusion_temporal_routeB/build/build_firstframe_head.log
```

### Required TensorRT plugin libraries

Put this file in:

```text
build/
```

Required:

```text
build/libmapdiffusion_msda.so
```

Put these files in:

```text
build/plugins/
```

Required:

```text
build/plugins/libmmcv_dcnv2_trt.so
build/plugins/libbevformer_tsa_trt.so
build/plugins/libbevformer_sca_trt.so
```

### `.engine` files

This runtime uses TensorRT serialized engine files with the `.plan` extension.

There are **no separate `.engine` files required** for the current working pipeline. If you rename `.plan` to `.engine`, you must also update the paths in:

```text
ports/mapdiffusion_temporal_routeB/run/run_temporal_routeB_val_submission.py
```

Recommended: keep the `.plan` filenames exactly as listed above.

### Artifact folder checklist

After downloading, this command should succeed:

```bash
ls -lh model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan
ls -lh model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan
ls -lh model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan
ls -lh model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan
ls -lh model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan

ls -lh build/libmapdiffusion_msda.so
ls -lh build/plugins/libmmcv_dcnv2_trt.so
ls -lh build/plugins/libbevformer_tsa_trt.so
ls -lh build/plugins/libbevformer_sca_trt.so
```

---

## Engine overview

The runtime is split into several TensorRT engines so that the Temporal MapDiffusion pipeline can be executed end-to-end while keeping each exported subgraph manageable.

| Engine | File | Purpose | Main input | Main output |
|---|---|---|---|---|
| Engine A: Backbone + FPN | `camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan` | Extracts multi-scale image features from the 6 nuScenes camera images. Uses the camera backbone/FPN path exported from the trained model. | `img` with shape `[1, 6, 3, 480, 800]` | `feat0`, `feat1`, `feat2` |
| Engine B: BEVFormer encoder | `camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan` | Projects multi-view image features into BEV using BEVFormer-style temporal/spatial attention plugins. | `feat0`, `feat1`, `feat2`, `ego2img` | `bev_features` with shape `[1, 256, 50, 100]` |
| Engine D: StreamFusionNeck | `stream_fusion_neck.temporal87000.fp32.plan` | Fuses current BEV with pose-warped previous BEV to produce the temporal BEV used by the map head. | `prev_bev`, `curr_bev` | `fused_bev` |
| Engine C_first: first-frame MapDiffusion head | `mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan` | Runs the MapDiffusion head for the first frame of a scene, where previous query memory is invalid or unavailable. | `fused_bev`, `query_coords`, `timestep` | `query_feat`, `line_preds`, `cls_logits` |
| Engine C_temporal: temporal MapDiffusion head | `mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan` | Runs the temporal MapDiffusion head for later frames using previous-frame query feature memory. | `fused_bev`, `query_coords`, `timestep`, `prev_query_feat` | `query_feat`, `line_preds`, `cls_logits` |

Runtime state:

```text
prev_bev_state         # previous fused BEV, pose-warped before Engine D
prev_query_feat_state  # previous frame final query features, used by C_temporal
```

Both states reset at scene boundaries.

---

## Output demo video
https://github.com/user-attachments/assets/54e35321-4621-468b-adc9-86a76a6df837
```text
Also in : artifacts/trt_warpbev_dataset_order_idx1566_1587_cv2.mp4
```

---

## External dependencies

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

## Setup

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

## Smoke test: 2-sample inference

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

## Full validation inference

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

## Chunked full validation

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

## Evaluate submission

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

## Important runtime notes

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

## Troubleshooting

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

## Minimal command sequence

```bash
git clone https://github.com/manavgoel472003/CUDA-MapDiffusion-Temporal.git
cd CUDA-MapDiffusion-Temporal

# Download artifacts from <YOUR LINK> and place them under:
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
