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
TRACE = ROOT / "model/mapdiffusion_temporal_routeB/pt_sampler_trace_sample0"
OUT = ROOT / "model/mapdiffusion_temporal_routeB/trt_replay_on_pt_trace_sample0"
OUT.mkdir(parents=True, exist_ok=True)

logger = trt.Logger(trt.Logger.WARNING)
load_plugins()
C = TrtModule("TemporalEngineCReplay", load_engine(ENGINE_C, logger))

bev = np.ascontiguousarray(np.load(TRACE / "bev_features.npy").astype(np.float32))
prev_qfeat = np.ascontiguousarray(np.load(TRACE / "prev_query_feat.npy").astype(np.float32))

print("bev", bev.shape, float(bev.min()), float(bev.max()), float(bev.mean()))
print("prev_qfeat", prev_qfeat.shape, float(prev_qfeat.min()), float(prev_qfeat.max()), float(prev_qfeat.mean()))

for step in range(5):
    q_path = TRACE / f"step_{step:02d}_input_query_coords.npy"
    t_path = TRACE / f"step_{step:02d}_input_timestep.npy"
    if not q_path.exists():
        break

    q = np.ascontiguousarray(np.load(q_path).astype(np.float32))
    t = np.ascontiguousarray(np.load(t_path).astype(np.float32))

    C.bind_host_input("bev_features", bev)
    C.bind_host_input("query_coords", q)
    C.bind_host_input("timestep", t)
    C.bind_host_input("prev_query_feat", prev_qfeat)
    C.allocate_outputs()
    C.execute()
    out = C.copy_outputs_to_host()

    print("=" * 100)
    print("step", step, "timestep", t.tolist())
    for k, v in out.items():
        print("trt_" + k, v.shape, float(v.min()), float(v.max()), float(v.mean()))
        np.save(OUT / f"step_{step:02d}_trt_{k}.npy", v.astype(np.float32))

print("saved:", OUT)
