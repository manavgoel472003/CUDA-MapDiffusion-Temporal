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

ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"
TRACE_DIR = ROOT / "model/mapdiffusion_temporal_routeB/pytorch_trace_sample0"
OUT_DIR = ROOT / "model/mapdiffusion_temporal_routeB/parity_single_step_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logger = trt.Logger(trt.Logger.WARNING)

load_plugins()
engine = load_engine(ENGINE_C, logger)
C = TrtModule("TemporalEngineCParity", engine)

bev = np.ascontiguousarray(np.load(TRACE_DIR / "bev_features.npy").astype(np.float32))
query = np.ascontiguousarray(np.load(TRACE_DIR / "step_00_query_coords.npy").astype(np.float32))
timestep = np.ascontiguousarray(np.array([1000.0], dtype=np.float32))
prev_query_feat = np.ascontiguousarray(np.zeros((1, 100, 512), dtype=np.float32))

print("bev", bev.shape, bev.dtype, float(bev.min()), float(bev.max()), float(bev.mean()))
print("query", query.shape, query.dtype, float(query.min()), float(query.max()), float(query.mean()))
print("timestep", timestep.shape, timestep.dtype, timestep.tolist())
print("prev_query_feat", prev_query_feat.shape, prev_query_feat.dtype)

C.bind_host_input("bev_features", bev)
C.bind_host_input("query_coords", query)
C.bind_host_input("timestep", timestep)
C.bind_host_input("prev_query_feat", prev_query_feat)
C.allocate_outputs()

print("executing C...")
C.execute()
out = C.copy_outputs_to_host()

for k, v in out.items():
    print("trt_" + k, v.shape, v.dtype, float(v.min()), float(v.max()), float(v.mean()))
    np.save(OUT_DIR / f"trt_{k}.npy", v)

print("saved:", OUT_DIR)
