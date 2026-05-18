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
echo "ONNX: $base/camera.bevformer.parse_probe_no_dcn.onnx"
ls -lh $base/camera.bevformer.parse_probe_no_dcn.onnx

$TRTEXEC \
  --onnx=$base/camera.bevformer.parse_probe_no_dcn.onnx \
  --fp16 \
  --saveEngine=$base/build/camera.bevformer.parse_probe_no_dcn.plan \
  --memPoolSize=workspace:8192 \
  --verbose \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --exportLayerInfo=$base/build/camera.bevformer.parse_probe_no_dcn.json \
  > $base/build/camera.bevformer.parse_probe_no_dcn.log 2>&1

echo "built: $base/build/camera.bevformer.parse_probe_no_dcn.plan"
ls -lh $base/build/camera.bevformer.parse_probe_no_dcn.plan
