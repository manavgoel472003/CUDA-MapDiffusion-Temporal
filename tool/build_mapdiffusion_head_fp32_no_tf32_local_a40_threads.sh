#!/bin/bash
set -eo pipefail

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

source tool/environment.sh

# Use more CPU threads locally.
# Try 16 first. If the machine is quiet, try 32 later.
export TRT_CPU_THREADS=${TRT_CPU_THREADS:-16}

export OMP_NUM_THREADS=$TRT_CPU_THREADS
export MKL_NUM_THREADS=$TRT_CPU_THREADS
export OPENBLAS_NUM_THREADS=$TRT_CPU_THREADS
export NUMEXPR_NUM_THREADS=$TRT_CPU_THREADS

# CUDA 11 runtime for TensorRT 8.5.3.1.
# This worked in your interactive shell.
export CUDA11_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/11.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CUDA11_LIB:$TensorRT_Lib:$LD_LIBRARY_PATH

base=model/mapdiffusion_routeA
mkdir -p $base/build

TRTEXEC=$TensorRT_Bin/trtexec
MSDA_PLUGIN=build/libmapdiffusion_msda.so

ONNX=$base/mapdiffusion.head.onnx
PLAN=$base/build/mapdiffusion.head.fp32.no_tf32.local_a40.plan
LOG=$base/build/mapdiffusion.head.fp32.no_tf32.local_a40.log

echo "============================================================"
echo "Local A40 no-TF32 build"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "TRT_CPU_THREADS=$TRT_CPU_THREADS"
echo "Workspace: 32768 MiB"
echo "============================================================"

nvidia-smi || true
lscpu | egrep "Model name|CPU\\(s\\)|Thread|Core|Socket" || true

echo "============================================================"
echo "Library check"
echo "============================================================"
ldd "$TRTEXEC" | grep -E "cudart|nvinfer|not found" || true

if ldd "$TRTEXEC" | grep -q "not found"; then
    echo "ERROR: trtexec has missing shared libraries."
    ldd "$TRTEXEC" | grep "not found" || true
    exit 1
fi

ls -lh "$ONNX"
ls -lh "$MSDA_PLUGIN"

echo "============================================================"
echo "Build"
echo "============================================================"

# Bind to first N logical CPUs. This does not force TensorRT to use all,
# but prevents it from bouncing across the whole machine.
CPU_LAST=$((TRT_CPU_THREADS - 1))

taskset -c 0-${CPU_LAST} \
$TRTEXEC \
  --plugins=$MSDA_PLUGIN \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:32768 \
  --buildOnly \
  --noTF32 \
  --minTiming=1 \
  --avgTiming=1 \
  > "$LOG" 2>&1

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
tail -160 "$LOG"
ls -lh "$PLAN"
