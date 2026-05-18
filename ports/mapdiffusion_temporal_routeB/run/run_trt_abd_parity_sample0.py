import sys
from pathlib import Path
import numpy as np
import tensorrt as trt

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
sys.path.insert(0, str(ROOT))

from ports.mapdiffusion_temporal_routeB.run.run_e2e_temporal_routeB_vis import (
    TrtModule,
    load_plugins,
    load_engine,
)

TRACE_DIR = ROOT / "model/mapdiffusion_temporal_routeB/pytorch_trace_sample0"
OUT_DIR = ROOT / "model/mapdiffusion_temporal_routeB/parity_abd_sample0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENGINE_A = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan"
ENGINE_B = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan"
ENGINE_D = ROOT / "model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan"

logger = trt.Logger(trt.Logger.WARNING)
load_plugins()

A = TrtModule("EngineA_BackboneFPN", load_engine(ENGINE_A, logger))
B = TrtModule("EngineB_BEVFormerEncoder", load_engine(ENGINE_B, logger))
D = TrtModule("EngineD_StreamFusionNeck", load_engine(ENGINE_D, logger))

img = np.ascontiguousarray(np.load(TRACE_DIR / "img.npy").astype(np.float32))
ego2img = np.ascontiguousarray(np.load(TRACE_DIR / "ego2img.npy").astype(np.float32))

print("img", img.shape, img.dtype, float(img.min()), float(img.max()), float(img.mean()))
print("ego2img", ego2img.shape, ego2img.dtype, float(ego2img.min()), float(ego2img.max()), float(ego2img.mean()))

A.bind_host_input("img", img)
A.allocate_outputs()

B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
B.bind_host_input("ego2img", ego2img)
B.allocate_outputs()

# First frame convention used by our runner: prev_bev = current raw_bev.
D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
D.allocate_outputs()

A.execute()
B.execute()
D.execute()

a_out = A.copy_outputs_to_host()
b_out = B.copy_outputs_to_host()
d_out = D.copy_outputs_to_host()

for k, v in a_out.items():
    v = np.ascontiguousarray(v.astype(np.float32))
    print("A", k, v.shape, float(v.min()), float(v.max()), float(v.mean()))
    np.save(OUT_DIR / f"trt_{k}.npy", v)

raw = np.ascontiguousarray(b_out["bev_features"].astype(np.float32))
fused = np.ascontiguousarray(d_out["fused_bev"].astype(np.float32))

print("B raw_bev", raw.shape, float(raw.min()), float(raw.max()), float(raw.mean()))
print("D fused_bev", fused.shape, float(fused.min()), float(fused.max()), float(fused.mean()))

np.save(OUT_DIR / "trt_raw_bev.npy", raw)
np.save(OUT_DIR / "trt_fused_bev.npy", fused)
print("saved:", OUT_DIR)
