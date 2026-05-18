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

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $base/camera.bevformer.onnx"
ls -lh $base/camera.bevformer.onnx

$TRTEXEC \
  --onnx=$base/camera.bevformer.onnx \
  --fp16 \
  --saveEngine=$base/build/camera.bevformer.plan \
  --memPoolSize=workspace:8192 \
  --verbose \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --exportLayerInfo=$base/build/camera.bevformer.json \
  > $base/build/camera.bevformer.log 2>&1

echo "built: $base/build/camera.bevformer.plan"
ls -lh $base/build/camera.bevformer.plan
