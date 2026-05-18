#!/bin/bash
set -eo pipefail

. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit 1
fi

base=model/mapdiffusion_routeA
mkdir -p $base/build

TRTEXEC=$TensorRT_Bin/trtexec
MSDA_PLUGIN=build/libmapdiffusion_msda.so

ONNX=$base/mapdiffusion.head.onnx
PLAN=$base/build/mapdiffusion.head.fp32.no_tf32.plan
LOG=$base/build/mapdiffusion.head.fp32.no_tf32.log

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $ONNX"
echo "Plugin: $MSDA_PLUGIN"
echo "Workspace: 24576 MiB"
echo "GPU memory before build:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv || true

ls -lh "$ONNX"
ls -lh "$MSDA_PLUGIN"

$TRTEXEC \
  --plugins=$MSDA_PLUGIN \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:24576 \
  --buildOnly \
  --noTF32 \
  --minTiming=1 \
  --avgTiming=1 \
  > $LOG 2>&1

echo "built: $PLAN"
ls -lh "$PLAN"
