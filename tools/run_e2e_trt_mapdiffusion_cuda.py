import os
import ctypes
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")

ENGINE_A = ROOT / "model/cuda_bevformer/build/camera.backbone_fpn.dcnv2.fp32.plan"
ENGINE_B = ROOT / "model/cuda_bevformer/build/camera.bevformer_encoder.tsa_sca_plugin.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_routeA/build/mapdiffusion.head.fp32.plan"

PLUGIN_DCN = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MD_MSDA = ROOT / "build/libmapdiffusion_msda.so"

OUT_DIR = ROOT / "model/cuda_bevformer/e2e_pipeline_inmemory"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def check_cuda(err, msg="CUDA call failed"):
    # cuda-python sometimes returns cudaError_t directly and sometimes
    # returns a tuple like (cudaError_t.cudaSuccess,).
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
    err = cudart.cudaMemcpy(
        ptr,
        arr,
        arr.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
    )
    check_cuda(err, "cudaMemcpy H2D failed")


def memcpy_d2h(arr, ptr):
    err = cudart.cudaMemcpy(
        arr,
        ptr,
        arr.nbytes,
        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
    )
    check_cuda(err, "cudaMemcpy D2H failed")


def load_plugins():
    for p in [PLUGIN_DCN, PLUGIN_TSA, PLUGIN_SCA, PLUGIN_MD_MSDA]:
        if not p.exists():
            raise FileNotFoundError(f"Missing plugin: {p}")
        ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
        print(f"[plugin] loaded: {p}")


def load_engine(path, logger):
    if not path.exists():
        raise FileNotFoundError(path)

    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError(f"Failed to deserialize: {path}")

    print(f"[engine] loaded: {path}")
    return engine


class TrtModule:
    def __init__(self, name, engine):
        self.name = name
        self.engine = engine
        self.ctx = engine.create_execution_context()
        self.bindings = [0] * engine.num_bindings
        self.owned_ptrs = {}
        self.output_host = {}

        print(f"\n[{self.name}] bindings")
        for i in range(engine.num_bindings):
            bname = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = tuple(engine.get_binding_shape(i))
            print(f"  {'IN ' if is_input else 'OUT'} {i}: {bname} {shape} {dtype}")

    def set_input_shape(self, name, shape):
        idx = self.engine.get_binding_index(name)
        self.ctx.set_binding_shape(idx, tuple(shape))

    def bind_device_input(self, name, ptr, shape):
        idx = self.engine.get_binding_index(name)
        self.ctx.set_binding_shape(idx, tuple(shape))
        self.bindings[idx] = int(ptr)

    def bind_host_input(self, name, arr):
        arr = np.ascontiguousarray(arr.astype(trt.nptype(self.engine.get_binding_dtype(
            self.engine.get_binding_index(name)
        )), copy=False))

        idx = self.engine.get_binding_index(name)
        self.ctx.set_binding_shape(idx, arr.shape)

        ptr = cuda_malloc(arr.nbytes)
        memcpy_h2d(ptr, arr)

        self.owned_ptrs[f"input:{name}"] = ptr
        self.bindings[idx] = int(ptr)

    def allocate_outputs(self):
        outputs = {}
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                continue

            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.ctx.get_binding_shape(i))

            if any(d < 0 for d in shape):
                raise RuntimeError(f"{self.name}: dynamic output shape unresolved for {name}: {shape}")

            host = np.empty(shape, dtype=dtype)
            ptr = cuda_malloc(host.nbytes)

            self.owned_ptrs[f"output:{name}"] = ptr
            self.output_host[name] = host
            self.bindings[i] = int(ptr)
            outputs[name] = (ptr, host, shape)

        return outputs

    def execute(self):
        ok = self.ctx.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError(f"{self.name}: execute_v2 failed")

    def copy_outputs_to_host(self):
        out = {}
        for name, host in self.output_host.items():
            ptr = self.owned_ptrs[f"output:{name}"]
            memcpy_d2h(host, ptr)
            out[name] = host
        return out

    def output_ptr(self, name):
        return self.owned_ptrs[f"output:{name}"]

    def output_shape(self, name):
        return tuple(self.output_host[name].shape)


def _load_env_array(env_name, shape, dtype=np.float32):
    path = os.environ.get(env_name, "")
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{env_name} points to missing file: {path}")

    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        arr = np.fromfile(path, dtype=dtype)

    arr = np.asarray(arr, dtype=dtype).reshape(shape)
    arr = np.ascontiguousarray(arr)

    print(f"[inputs] loaded {env_name}: {path} {arr.shape} min/max {arr.min():.6f} {arr.max():.6f}")
    return arr


def create_inputs():
    img = _load_env_array("E2E_IMG", (1, 6, 3, 480, 800), np.float32)
    ego2img = _load_env_array("E2E_EGO2IMG", (1, 6, 4, 4), np.float32)

    if img is None:
        rng = np.random.default_rng(123)
        img = rng.normal(0.0, 1.0, size=(1, 6, 3, 480, 800)).astype(np.float32)
        print(f"[inputs] using deterministic random image {img.shape} min/max {img.min():.6f} {img.max():.6f}")

    if ego2img is None:
        ego2img = np.zeros((1, 6, 4, 4), dtype=np.float32)
        for i in range(6):
            ego2img[0, i] = np.eye(4, dtype=np.float32)
        print(f"[inputs] using identity ego2img {ego2img.shape} min/max {ego2img.min():.6f} {ego2img.max():.6f}")

    query_coords = _load_env_array("E2E_QUERY_COORDS", (1, 100, 20, 2), np.float32)
    if query_coords is None:
        rng = np.random.default_rng(456)
        query_coords = rng.uniform(0.0, 1.0, size=(1, 100, 20, 2)).astype(np.float32)
        print(f"[inputs] using deterministic random query_coords {query_coords.shape} min/max {query_coords.min():.6f} {query_coords.max():.6f}")

    return img, ego2img, query_coords

def make_event():
    err, ev = cudart.cudaEventCreate()
    check_cuda(err, "cudaEventCreate failed")
    return ev


def time_loop(fn, warmup=20, iters=200):
    start = make_event()
    end = make_event()

    for _ in range(warmup):
        fn()

    check_cuda(cudart.cudaDeviceSynchronize()[0], "sync failed")

    err = cudart.cudaEventRecord(start, 0)
    check_cuda(err, "record start failed")

    for _ in range(iters):
        fn()

    err = cudart.cudaEventRecord(end, 0)
    check_cuda(err, "record end failed")

    err = cudart.cudaEventSynchronize(end)
    check_cuda(err, "event sync failed")

    err, ms = cudart.cudaEventElapsedTime(start, end)
    check_cuda(err, "elapsed failed")

    return ms / iters


def main():
    logger = trt.Logger(trt.Logger.WARNING)

    load_plugins()

    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_c = load_engine(ENGINE_C, logger)

    img, ego2img, query_coords = create_inputs()
    timestep = np.array([0.0], dtype=np.float32)

    A = TrtModule("EngineA_BackboneFPN", engine_a)
    B = TrtModule("EngineB_BEVFormerEncoder", engine_b)
    C = TrtModule("EngineC_MapDiffusionHead", engine_c)

    # Engine A: img -> feat0/feat1/feat2
    A.bind_host_input("img", img)
    A.allocate_outputs()

    # Engine B: use Engine A output device pointers directly.
    B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
    B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
    B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
    B.bind_host_input("ego2img", ego2img)
    B.allocate_outputs()

    # Engine C: use Engine B BEV output device pointer directly.
    C.bind_device_input("bev_features", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    C.bind_host_input("query_coords", query_coords)
    C.bind_host_input("timestep", timestep)
    C.allocate_outputs()

    def run_once():
        A.execute()
        B.execute()
        C.execute()

    print("\n============================================================")
    print("Smoke run")
    print("============================================================")
    run_once()
    check_cuda(cudart.cudaDeviceSynchronize()[0], "final sync failed")

    out_a = A.copy_outputs_to_host()
    out_b = B.copy_outputs_to_host()
    out_c = C.copy_outputs_to_host()

    for prefix, outputs in [("trt_", out_a), ("trt_", out_b), ("trt_", out_c)]:
        for name, arr in outputs.items():
            np.save(OUT_DIR / f"{prefix}{name}.npy", arr)
            print(prefix + name, arr.shape, arr.dtype, float(arr.min()), float(arr.max()), float(arr.mean()))

    # Also save the exact inputs used by TRT so PyTorch reference uses identical tensors.
    np.save(OUT_DIR / "input_img.npy", img)
    np.save(OUT_DIR / "input_ego2img.npy", ego2img)
    np.save(OUT_DIR / "input_query_coords.npy", query_coords)
    np.save(OUT_DIR / "input_timestep.npy", timestep)

    print("\n============================================================")
    print("Latency: full in-memory A -> B -> C")
    print("============================================================")

    ms_one_head = time_loop(run_once, warmup=20, iters=200)
    print(f"Full pipeline one-head-call latency: {ms_one_head:.3f} ms")
    print(f"Approx 5-step MapDiffusion latency: A+B+5C is NOT measured by this function yet.")
    print("This measured A+B+C once, using device pointers between engines.")

    print("\nDONE")
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
