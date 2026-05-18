from pathlib import Path

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")

ENGINE_A = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan"
ENGINE_B = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan"
ENGINE_D = ROOT / "model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"

PLUGIN_DCNV2 = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MSDA = ROOT / "build/libmapdiffusion_msda.so"
