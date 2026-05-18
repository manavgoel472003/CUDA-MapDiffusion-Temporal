#!/bin/bash
set -eo pipefail

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion
source tool/environment.sh

export CUDA11_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/11.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CUDA11_LIB:$TensorRT_Lib:$LD_LIBRARY_PATH

base=model/mapdiffusion_routeA
mkdir -p $base/build

TRTEXEC=$TensorRT_Bin/trtexec
MSDA_PLUGIN=build/libmapdiffusion_msda.so

ONNX=$base/mapdiffusion.head.onnx
PLAN=$base/build/mapdiffusion.head.fp32.no_tf32.local_a40.fast.plan
LOG=$base/build/mapdiffusion.head.fp32.no_tf32.local_a40.fast.log

echo "============================================================"
echo "Fast local A40 no-TF32 build"
echo "Start: $(date)"
echo "TRTEXEC: $TRTEXEC"
echo "ONNX: $ONNX"
echo "PLAN: $PLAN"
echo "LOG: $LOG"
echo "============================================================"

ldd "$TRTEXEC" | grep -E "cudart|nvinfer|not found" || true
nvidia-smi || true

$TRTEXEC \
  --plugins=$MSDA_PLUGIN \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:32768 \
  --buildOnly \
  --noTF32 \
  --heuristic \
  --minTiming=1 \
  --avgTiming=1 \
  > "$LOG" 2>&1

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"

tail -120 "$LOG"
ls -lh "$PLAN"
