#!/usr/bin/env bash
set -euo pipefail

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

source tool/environment.sh

export CUDA11_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda/11.8/targets/x86_64-linux/lib
export CUBLAS_LIB=/opt/fs/atipa/local/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/math_libs/11.8/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$CUDA11_LIB:$CUBLAS_LIB:$TensorRT_Lib:${LD_LIBRARY_PATH:-}

export TEMPORAL_CONFIG=/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/temporal_config.py
export TEMPORAL_CKPT=/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth

mkdir -p model/mapdiffusion_temporal_routeB/{onnx,build}

echo "========== Export Engine A =========="
python ports/mapdiffusion_temporal_routeB/export/export_temporal_camera_backbone_fpn_tensorio.py \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/export_A.log

echo "========== Build Engine A FP32 =========="
$TensorRT_Bin/trtexec \
  --plugins=build/plugins/libmmcv_dcnv2_trt.so \
  --onnx=model/mapdiffusion_temporal_routeB/onnx/camera.backbone_fpn.temporal87000.onnx \
  --saveEngine=model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan \
  --memPoolSize=workspace:8192 \
  --buildOnly \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/build_A_fp32.log

echo "========== Export Engine B =========="
python ports/mapdiffusion_temporal_routeB/export/export_temporal_camera_bevformer_encoder_sca_plugin_tensorio.py \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/export_B.log

echo "========== Build Engine B =========="
$TensorRT_Bin/trtexec \
  --plugins=build/plugins/libbevformer_tsa_trt.so \
  --plugins=build/plugins/libbevformer_sca_trt.so \
  --onnx=model/mapdiffusion_temporal_routeB/onnx/camera.bevformer_encoder.temporal87000.sca_plugin.onnx \
  --saveEngine=model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan \
  --memPoolSize=workspace:8192 \
  --buildOnly \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/build_B.log

echo "========== Export Engine D =========="
export ONNX_OUT=/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/onnx/stream_fusion_neck.temporal87000.fp32.onnx
python ports/mapdiffusion_temporal_routeB/export/export_temporal_stream_fusion_neck_onnx.py \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/export_D.log
unset ONNX_OUT

echo "========== Build Engine D =========="
$TensorRT_Bin/trtexec \
  --onnx=model/mapdiffusion_temporal_routeB/onnx/stream_fusion_neck.temporal87000.fp32.onnx \
  --saveEngine=model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan \
  --memPoolSize=workspace:4096 \
  --buildOnly \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/build_D.log

echo "========== Export Engine C =========="
export ONNX_OUT=/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion/model/mapdiffusion_temporal_routeB/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.onnx
python ports/mapdiffusion_temporal_routeB/export/export_temporal_head_onnx.py \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/export_C.log
unset ONNX_OUT

echo "========== Build Engine C =========="
$TensorRT_Bin/trtexec \
  --plugins=build/libmapdiffusion_msda.so \
  --onnx=model/mapdiffusion_temporal_routeB/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.onnx \
  --saveEngine=model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan \
  --memPoolSize=workspace:8192 \
  --buildOnly \
  2>&1 | tee ports/mapdiffusion_temporal_routeB/logs/build_C.log

echo "========== Final plans =========="
ls -lh \
  model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan \
  model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan \
  model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan \
  model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan
