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

TSA_PLUGIN=build/plugins/libbevformer_tsa_trt.so
SCA_PLUGIN=build/plugins/libbevformer_sca_trt.so

ONNX=$base/camera.bevformer_encoder.sca_plugin.onnx
PLAN=$base/build/camera.bevformer_encoder.tsa_sca_plugin.fp32.plan
LOG=$base/build/camera.bevformer_encoder.tsa_sca_plugin.fp32.log

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $ONNX"
echo "TSA plugin: $TSA_PLUGIN"
echo "SCA plugin: $SCA_PLUGIN"

ls -lh "$ONNX"
ls -lh "$TSA_PLUGIN"
ls -lh "$SCA_PLUGIN"

$TRTEXEC \
  --plugins=$TSA_PLUGIN \
  --plugins=$SCA_PLUGIN \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:1024 \
  --buildOnly \
  --minTiming=1 \
  --avgTiming=1 \
  > $LOG 2>&1

echo "built: $PLAN"
ls -lh "$PLAN"
