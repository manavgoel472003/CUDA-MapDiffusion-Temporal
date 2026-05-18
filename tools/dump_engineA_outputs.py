#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
sys.path.insert(0, str(ROOT / "tools"))

from run_e2e_trt_mapdiffusion_5step_vis import TrtModule, load_plugins, load_engine

IMG = ROOT / "model/cuda_bevformer/real_sample_inputs/img.npy"

def main():
    if len(sys.argv) != 3:
        print("Usage: python tools/dump_engineA_outputs.py <engine.plan> <outdir>")
        sys.exit(1)

    plan = ROOT / sys.argv[1]
    outdir = ROOT / sys.argv[2]
    outdir.mkdir(parents=True, exist_ok=True)

    print("engine:", plan)
    print("outdir:", outdir)

    load_plugins()

    logger = trt.Logger(trt.Logger.WARNING)
    try:
        engine = load_engine(plan, logger)
    except TypeError:
        engine = load_engine(plan)

    A = TrtModule("EngineA_Dump", engine)

    img = np.ascontiguousarray(np.load(IMG).astype(np.float32))
    print("img:", img.shape, img.dtype, float(img.min()), float(img.max()), float(img.mean()))

    A.bind_host_input("img", img)
    A.allocate_outputs()
    A.execute()

    out = A.copy_outputs_to_host()
    for name, arr in out.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        np.save(outdir / f"{name}.npy", arr)
        print(name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()), float(arr.mean()))

if __name__ == "__main__":
    main()
