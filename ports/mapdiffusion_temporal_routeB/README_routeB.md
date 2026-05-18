# MapDiffusion Temporal Route B TensorRT Port

Checkpoint:

/home/018198687/Mapping/mapdiffusion/work_dirs/mapdiffusion_temporal_scratch_32e_start1_stronger_loss/iter_87000.pth

Config:

model/mapdiffusion_temporal_routeB/temporal_config.py

## Engine flow

A: img -> feat0, feat1, feat2

B: feat0, feat1, feat2, ego2img -> raw_bev

D: raw_bev, raw_bev -> fused_bev

C: fused_bev, query_coords, timestep, prev_query_feat -> line_preds, cls_logits, query_feat

## Engine files

A:
model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan

B:
model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan

D:
model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan

C:
model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan

## Plugins used

DCNv2:
build/plugins/libmmcv_dcnv2_trt.so

BEVFormer TSA:
build/plugins/libbevformer_tsa_trt.so

BEVFormer SCA:
build/plugins/libbevformer_sca_trt.so

MapDiffusion MSDA:
build/libmapdiffusion_msda.so

## Why C uses ManualTemporalQueryFusion

The original temporal head uses PyTorch nn.MultiheadAttention. Its ONNX export created dynamic reshape patterns like [100, -1, 512], which TensorRT 8.5 could not resolve.

Route B replaces only the export-time temporal_query_fusion module with a fixed-shape Q/K/V attention implementation using the same trained weights.

## Build all engines

cd /home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion

ports/mapdiffusion_temporal_routeB/export/build_all_routeB_plans.sh

## First-frame temporal memory

For first frame:

prev_query_feat = zeros(1, 100, 512)

For sequence mode:

prev_query_feat = previous frame query_feat

## Current complete plan set

A, B, C, D should exist here:

model/mapdiffusion_temporal_routeB/build/

Check with:

ls -lh \
  model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan \
  model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan \
  model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan \
  model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan
