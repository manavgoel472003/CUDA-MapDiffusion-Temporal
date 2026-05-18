import os
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as e:
    raise RuntimeError(
        "pycuda is required for this runner. Try: python -c 'import pycuda.driver'"
    ) from e


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")

ENGINE_A = ROOT / "model/cuda_bevformer/build/camera.backbone_fpn.dcnv2.fp32.plan"
ENGINE_B = ROOT / "model/cuda_bevformer/build/camera.bevformer_encoder.tsa_sca_plugin.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_routeA/build/mapdiffusion.head.fp32.plan"

PLUGIN_DCN = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MD_MSDA = ROOT / "build/libmapdiffusion_msda.so"

OUT_DIR = ROOT / "model/cuda_bevformer/e2e_pipeline_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_plugins():
    for p in [PLUGIN_DCN, PLUGIN_TSA, PLUGIN_SCA, PLUGIN_MD_MSDA]:
        if not p.exists():
            raise FileNotFoundError(f"Missing plugin: {p}")
        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        print(f"[plugin] loaded: {p}")


def load_engine(path: Path, logger):
    if not path.exists():
        raise FileNotFoundError(path)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize engine: {path}")
    print(f"[engine] loaded: {path}")
    return engine


def binding_names(engine):
    return [engine.get_binding_name(i) for i in range(engine.num_bindings)]


def dtype_to_np(dtype):
    return trt.nptype(dtype)


def allocate_and_run(engine, inputs, label):
    context = engine.create_execution_context()

    # Set static/dynamic shapes if needed.
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if engine.binding_is_input(i):
            arr = inputs[name]
            context.set_binding_shape(i, arr.shape)

    bindings = [None] * engine.num_bindings
    device_allocs = {}
    host_outputs = {}

    # Allocate inputs and outputs.
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = dtype_to_np(engine.get_binding_dtype(i))

        if engine.binding_is_input(i):
            arr = np.ascontiguousarray(inputs[name].astype(dtype, copy=False))
            dptr = cuda.mem_alloc(arr.nbytes)
            cuda.memcpy_htod(dptr, arr)
            bindings[i] = int(dptr)
            device_allocs[name] = dptr
            print(f"[{label}] input  {name:15s} {arr.shape} {arr.dtype} min/max {arr.min():.6f} {arr.max():.6f}")
        else:
            shape = tuple(context.get_binding_shape(i))
            arr = np.empty(shape, dtype=dtype)
            dptr = cuda.mem_alloc(arr.nbytes)
            bindings[i] = int(dptr)
            device_allocs[name] = dptr
            host_outputs[name] = arr
            print(f"[{label}] output {name:15s} {arr.shape} {arr.dtype}")

    ok = context.execute_v2(bindings)
    if not ok:
        raise RuntimeError(f"{label}: TensorRT execute_v2 failed")

    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if not engine.binding_is_input(i):
            cuda.memcpy_dtoh(host_outputs[name], device_allocs[name])
            arr = host_outputs[name]
            print(f"[{label}] got    {name:15s} {arr.shape} {arr.dtype} min/max {arr.min():.6f} {arr.max():.6f}")

    return host_outputs


def make_inputs():
    # Prefer real parity inputs if available; otherwise use deterministic random smoke inputs.
    rng = np.random.default_rng(123)

    img_npy = OUT_DIR / "img.npy"
    ego_npy = OUT_DIR / "ego2img.npy"

    if img_npy.exists() and ego_npy.exists():
        img = np.load(img_npy).astype(np.float32)
        ego2img = np.load(ego_npy).astype(np.float32)
        print("[inputs] loaded existing img/ego2img from", OUT_DIR)
    else:
        print("[inputs] creating deterministic smoke img/ego2img")
        img = rng.normal(0, 1, size=(1, 6, 3, 480, 800)).astype(np.float32)

        # Identity-like camera matrices for smoke execution.
        ego2img = np.zeros((1, 6, 4, 4), dtype=np.float32)
        for c in range(6):
            ego2img[0, c] = np.eye(4, dtype=np.float32)

        np.save(img_npy, img)
        np.save(ego_npy, ego2img)

    # For one-step head smoke, use existing head parity query/timestep if available.
    parity = ROOT / "model/mapdiffusion_routeA/parity_head"
    qc_npy = parity / "query_coords.npy"
    ts_npy = parity / "timestep.npy"

    if qc_npy.exists() and ts_npy.exists():
        query_coords = np.load(qc_npy).astype(np.float32)
        timestep = np.load(ts_npy).astype(np.float32)
        print("[inputs] loaded query_coords/timestep from parity_head")
    else:
        print("[inputs] creating deterministic query_coords/timestep")
        query_coords = rng.uniform(0, 1, size=(1, 100, 20, 2)).astype(np.float32)
        timestep = np.array([1.0], dtype=np.float32)

    return img, ego2img, query_coords, timestep


def main():
    logger = trt.Logger(trt.Logger.INFO)

    load_plugins()

    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_c = load_engine(ENGINE_C, logger)

    print("[Engine A bindings]", binding_names(engine_a))
    print("[Engine B bindings]", binding_names(engine_b))
    print("[Engine C bindings]", binding_names(engine_c))

    img, ego2img, query_coords, timestep = make_inputs()

    out_a = allocate_and_run(
        engine_a,
        {"img": img},
        "EngineA-FPN",
    )

    out_b = allocate_and_run(
        engine_b,
        {
            "feat0": out_a["feat0"],
            "feat1": out_a["feat1"],
            "feat2": out_a["feat2"],
            "ego2img": ego2img,
        },
        "EngineB-BEVFormer",
    )

    out_c = allocate_and_run(
        engine_c,
        {
            "bev_features": out_b["bev_features"],
            "query_coords": query_coords,
            "timestep": timestep,
        },
        "EngineC-MapDiffHead",
    )

    for k, v in out_a.items():
        np.save(OUT_DIR / f"{k}.npy", v)
    for k, v in out_b.items():
        np.save(OUT_DIR / f"{k}.npy", v)
    for k, v in out_c.items():
        np.save(OUT_DIR / f"{k}.npy", v)

    print("=" * 80)
    print("E2E PIPELINE DONE")
    print("Saved outputs to:", OUT_DIR)
    print("cls_logits:", out_c["cls_logits"].shape, out_c["cls_logits"].min(), out_c["cls_logits"].max())
    print("line_preds:", out_c["line_preds"].shape, out_c["line_preds"].min(), out_c["line_preds"].max())
    print("=" * 80)


if __name__ == "__main__":
    main()
