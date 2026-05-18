#!/bin/bash
set -eo pipefail

. tool/environment.sh

if [ "$ConfigurationStatus" != "Success" ]; then
    echo "Exit due to configure failure."
    exit 1
fi

base=model/cuda_bevformer
mkdir -p $base/build

TRTEXEC=$TensorRT_Bin/trtexec

ONNX=$base/camera.bevformer_encoder.msda_trt.onnx
PLAN=$base/build/camera.bevformer_encoder.msda_trt.fp32.minlog.plan
LOG=$base/build/camera.bevformer_encoder.msda_trt.fp32.minlog.log

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $ONNX"

ls -lh "$ONNX"

$TRTEXEC \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:4096 \
  --buildOnly \
  --minTiming=1 \
  --avgTiming=1 \
  > $LOG 2>&1

echo "built: $PLAN"
ls -lh "$PLAN"
