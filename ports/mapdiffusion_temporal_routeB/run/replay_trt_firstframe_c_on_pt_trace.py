import sys
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
sys.path.insert(0, str(ROOT))

from ports.mapdiffusion_temporal_routeB.run.run_e2e_temporal_routeB_vis import (
    TrtModule,
    load_engine,
)

ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan"
TRACE = ROOT / "model/mapdiffusion_temporal_routeB/pt_sampler_trace_sample0"
OUT = ROOT / "model/mapdiffusion_temporal_routeB/trt_firstframe_replay_on_pt_trace_sample0"
OUT.mkdir(parents=True, exist_ok=True)

ctypes.CDLL(str(ROOT / "build/libmapdiffusion_msda.so"), mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.WARNING)
engine = load_engine(ENGINE_C, logger)
C = TrtModule("TemporalEngineCFirstFrameReplay", engine)

binding_names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
input_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if engine.binding_is_input(i)]

print("input bindings:", input_names)
print("all bindings:", binding_names)

bev = np.ascontiguousarray(np.load(TRACE / "bev_features.npy").astype(np.float32))
prev_qfeat = np.ascontiguousarray(np.load(TRACE / "prev_query_feat.npy").astype(np.float32))

for step in range(5):
    q_path = TRACE / f"step_{step:02d}_input_query_coords.npy"
    t_path = TRACE / f"step_{step:02d}_input_timestep.npy"
    if not q_path.exists():
        break

    q = np.ascontiguousarray(np.load(q_path).astype(np.float32))
    t = np.ascontiguousarray(np.load(t_path).astype(np.float32))

    if "bev_features" in input_names:
        C.bind_host_input("bev_features", bev)
    if "query_coords" in input_names:
        C.bind_host_input("query_coords", q)
    if "timestep" in input_names:
        C.bind_host_input("timestep", t)
    if "prev_query_feat" in input_names:
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
