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

ONNX=$base/camera.backbone_fpn.onnx
PLAN=$base/build/camera.backbone_fpn.dcnv2.fp32.plan
LOG=$base/build/camera.backbone_fpn.dcnv2.fp32.log

echo "Using trtexec: $TRTEXEC"
echo "ONNX: $ONNX"
echo "DCNv2 plugin: $DCN_PLUGIN"

ls -lh "$ONNX"
ls -lh "$DCN_PLUGIN"

$TRTEXEC \
  --plugins=$DCN_PLUGIN \
  --onnx=$ONNX \
  --saveEngine=$PLAN \
  --memPoolSize=workspace:4096 \
  --buildOnly \
  --minTiming=1 \
  --avgTiming=1 \
  --verbose \
  > $LOG 2>&1

echo "built: $PLAN"
ls -lh "$PLAN"
