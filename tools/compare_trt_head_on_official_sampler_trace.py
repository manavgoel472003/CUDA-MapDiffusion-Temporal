import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")
TRACE = Path("/home/018198687/Mapping/mapdiffusion/work_dirs/new_mapdiff_sampler_trace_idx0_v2")

ENGINE_C = ROOT / "model/mapdiffusion_routeA/build/mapdiffusion.head.fp32.plan"
PLUGIN_MD_MSDA = ROOT / "build/libmapdiffusion_msda.so"


def check_cuda(err, msg="CUDA call failed"):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{msg}: {err}")


def cuda_malloc(nbytes):
    err, ptr = cudart.cudaMalloc(nbytes)
    check_cuda(err, "cudaMalloc failed")
    return ptr


def memcpy_h2d(ptr, arr):
    arr = np.ascontiguousarray(arr)
    err = cudart.cudaMemcpy(ptr, arr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
    check_cuda(err, "cudaMemcpy H2D failed")


def memcpy_d2h(arr, ptr):
    err = cudart.cudaMemcpy(arr, ptr, arr.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
    check_cuda(err, "cudaMemcpy D2H failed")


def load_engine():
    ctypes.CDLL(str(PLUGIN_MD_MSDA), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(ENGINE_C, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Could not load engine: {ENGINE_C}")
    return engine


def metrics(a, b):
    a = a.astype(np.float32).reshape(-1)
    b = b.astype(np.float32).reshape(-1)
    d = a - b
    return {
        "elems": int(a.size),
        "max_abs": float(np.max(np.abs(d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "cosine": float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12)),
        "trt_min": float(a.min()),
        "trt_max": float(a.max()),
        "pt_min": float(b.min()),
        "pt_max": float(b.max()),
    }


def run_head(engine, bev_features, query_coords, timestep):
    ctx = engine.create_execution_context()
    bindings = [0] * engine.num_bindings
    owned = {}
    host_out = {}

    inputs = {
        "bev_features": bev_features.astype(np.float32),
        "query_coords": query_coords.astype(np.float32),
        "timestep": timestep.astype(np.float32),
    }

    for name, arr in inputs.items():
        idx = engine.get_binding_index(name)
        ctx.set_binding_shape(idx, arr.shape)
        ptr = cuda_malloc(arr.nbytes)
        memcpy_h2d(ptr, arr)
        bindings[idx] = int(ptr)
        owned[name] = ptr

    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            continue
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = tuple(ctx.get_binding_shape(i))
        arr = np.empty(shape, dtype=dtype)
        ptr = cuda_malloc(arr.nbytes)
        bindings[i] = int(ptr)
        owned[name] = ptr
        host_out[name] = arr

    ok = ctx.execute_v2(bindings)
    if not ok:
        raise RuntimeError("Engine C execute_v2 failed")

    check_cuda(cudart.cudaDeviceSynchronize()[0], "sync failed")

    for name, arr in host_out.items():
        memcpy_d2h(arr, owned[name])

    return host_out


def main():
    engine = load_engine()

    steps = sorted(TRACE.glob("step_*.npz"))
    if not steps:
        raise FileNotFoundError(f"No step_*.npz files found in {TRACE}")

    print("Engine:", ENGINE_C)
    print("Trace:", TRACE)
    print("Num official steps:", len(steps))

    for p in steps:
        data = np.load(p)

        bev = data["bev_features"].astype(np.float32)
        query = data["query_coords"].astype(np.float32)
        timestep = data["timestep"].astype(np.float32)

        pt_line = data["line_preds"].astype(np.float32).reshape(1, 100, 40)
        pt_cls = data["cls_logits"].astype(np.float32).reshape(1, 100, 3)

        out = run_head(engine, bev, query, timestep)

        trt_line = out["line_preds"].astype(np.float32).reshape(1, 100, 40)
        trt_cls = out["cls_logits"].astype(np.float32).reshape(1, 100, 3)

        print("=" * 100)
        print(p.name)
        print("timestep:", timestep.reshape(-1).tolist())

        print("line_preds")
        for k, v in metrics(trt_line, pt_line).items():
            print(f"  {k}: {v}")

        print("cls_logits")
        for k, v in metrics(trt_cls, pt_cls).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
