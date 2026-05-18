import os
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt

# Adjust if your wrapper class lives elsewhere.
from ports.mapdiffusion_temporal_routeB.run.run_temporal_routeB_val_submission import TrtModule as EngineWrapper

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
BUILD = ROOT / "model/mapdiffusion_temporal_routeB/build"

PT_ROOT = ROOT / "model/mapdiffusion_temporal_routeB/pytorch_working_trace_2"
OUT_ROOT = ROOT / "model/mapdiffusion_temporal_routeB/trt_replay_on_working_pt_inputs"

PLUGIN_SO = ROOT / "build/libmapdiffusion_msda.so"

C_TEMPORAL_PLAN = BUILD / "mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"
C_FIRST_PLAN = BUILD / "mapdiffusion.temporal_head.manual_tq.firstframe.opset13.fp32.plan"



def load_trt_engine(plan_path):
    plan_path = Path(plan_path)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(plan_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {plan_path}")
    return engine


def save(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr).astype(np.float32))


def load(path):
    return np.ascontiguousarray(np.load(path).astype(np.float32))


def main():
    print("[plugin] loading:", PLUGIN_SO)
    ctypes.CDLL(str(PLUGIN_SO), mode=ctypes.RTLD_GLOBAL)

    print("[engine] C temporal:", C_TEMPORAL_PLAN)
    print("[engine] C first:", C_FIRST_PLAN)

    C = EngineWrapper("EngineC_Temporal_Replay", load_trt_engine(C_TEMPORAL_PLAN))
    C_first = EngineWrapper("EngineC_First_Replay", load_trt_engine(C_FIRST_PLAN))

    for idx in [0, 1]:
        pt_dir = PT_ROOT / f"debug_e2e_idx{idx:06d}"
        out_dir = OUT_ROOT / f"debug_e2e_idx{idx:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 120)
        print("[idx]", idx)
        print("PT dir:", pt_dir)

        # idx0 in PyTorch has no previous memory, so use C_first.
        # idx1 in working PyTorch has prev_query_feat from idx0, so use C_temporal.
        use_first = idx == 0
        C_run = C_first if use_first else C

        for step in range(5):
            print("-" * 100)
            print("[step]", step, "engine", "C_first" if use_first else "C_temporal")

            bev = load(pt_dir / f"04_C_step{step:02d}_input_bev.npy")
            query = load(pt_dir / f"04_C_step{step:02d}_input_query_coords.npy")
            timestep = load(pt_dir / f"04_C_step{step:02d}_input_timestep.npy")

            C_run.bind_host_input("bev_features", bev)
            C_run.bind_host_input("query_coords", query)
            C_run.bind_host_input("timestep", timestep)

            if not use_first:
                prev_q = load(pt_dir / f"04_C_step{step:02d}_input_prev_query_feat.npy")
                C_run.bind_host_input("prev_query_feat", prev_q)

            C_run.allocate_outputs()
            C_run.execute()
            out = C_run.copy_outputs_to_host()

            line = np.ascontiguousarray(out["line_preds"].astype(np.float32))
            cls = np.ascontiguousarray(out["cls_logits"].astype(np.float32))
            qfeat = np.ascontiguousarray(out["query_feat"].astype(np.float32))

            save(out_dir / f"05_C_step{step:02d}_output_line_preds.npy", line)
            save(out_dir / f"05_C_step{step:02d}_output_cls_logits.npy", cls)
            save(out_dir / f"05_C_step{step:02d}_output_query_feat.npy", qfeat)

            save(out_dir / f"04_C_step{step:02d}_input_bev.npy", bev)
            save(out_dir / f"04_C_step{step:02d}_input_query_coords.npy", query)
            save(out_dir / f"04_C_step{step:02d}_input_timestep.npy", timestep)

            if not use_first:
                save(out_dir / f"04_C_step{step:02d}_input_prev_query_feat.npy", prev_q)
            else:
                save(out_dir / f"04_C_step{step:02d}_input_prev_query_feat.npy", np.zeros((1, 100, 512), dtype=np.float32))

            with open(out_dir / f"04_C_step{step:02d}_engine.txt", "w") as f:
                f.write("C_first\n" if use_first else "C_temporal\n")

            print("line", line.shape, float(line.min()), float(line.max()), float(line.mean()))
            print("cls ", cls.shape, float(cls.min()), float(cls.max()), float(cls.mean()))
            print("qft ", qfeat.shape, float(qfeat.min()), float(qfeat.max()), float(qfeat.mean()))

    print("[saved]", OUT_ROOT)


if __name__ == "__main__":
    main()
