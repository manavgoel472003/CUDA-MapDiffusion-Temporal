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
DCN_PLUGIN=build/plugins/libmmcv_dcnv2_trt.so

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $base/camera.bevformer.onnx"
echo "DCNv2 plugin: $DCN_PLUGIN"

ls -lh $base/camera.bevformer.onnx
ls -lh $DCN_PLUGIN

$TRTEXEC \
  --plugins=$DCN_PLUGIN \
  --onnx=$base/camera.bevformer.onnx \
  --saveEngine=$base/build/camera.bevformer.dcnv2.fp32.plan \
  --memPoolSize=workspace:8192 \
  --verbose \
  --dumpLayerInfo \
  --profilingVerbosity=detailed \
  --exportLayerInfo=$base/build/camera.bevformer.dcnv2.fp32.json \
  > $base/build/camera.bevformer.dcnv2.fp32.log 2>&1

echo "built: $base/build/camera.bevformer.dcnv2.fp32.plan"
ls -lh $base/build/camera.bevformer.dcnv2.fp32.plan
