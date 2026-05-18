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
SCA_PLUGIN=build/plugins/libbevformer_sca_trt.so

ONNX=$base/camera.bevformer_encoder.sca_plugin.no_msda_probe.onnx
PLAN=$base/build/camera.bevformer_encoder.sca_plugin.no_msda_probe.plan
LOG=$base/build/camera.bevformer_encoder.sca_plugin.no_msda_probe.log

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $ONNX"
echo "SCA plugin: $SCA_PLUGIN"

ls -lh "$ONNX"
ls -lh "$SCA_PLUGIN"

$TRTEXEC \
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
