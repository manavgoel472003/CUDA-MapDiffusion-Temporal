import os
import ctypes
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/018198687/Mapping/Lidar_AI_Solution/CUDA-BEVFusion")

ENGINE_A = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.backbone_fpn.temporal87000.dcnv2.fp32.im2colgemm.plan"
ENGINE_B = ROOT / "model/mapdiffusion_temporal_routeB/build/camera.bevformer_encoder.temporal87000.tsa_sca_plugin.fp32.plan"
ENGINE_C = ROOT / "model/mapdiffusion_temporal_routeB/build/mapdiffusion.temporal_head.manual_tq.presplit.opset13.fp32.plan"
ENGINE_D = ROOT / "model/mapdiffusion_temporal_routeB/build/stream_fusion_neck.temporal87000.fp32.plan"

PLUGIN_DCN = ROOT / "build/plugins/libmmcv_dcnv2_trt.so"
PLUGIN_TSA = ROOT / "build/plugins/libbevformer_tsa_trt.so"
PLUGIN_SCA = ROOT / "build/plugins/libbevformer_sca_trt.so"
PLUGIN_MD_MSDA = ROOT / "build/libmapdiffusion_msda.so"

OUT_DIR = ROOT / "model/cuda_bevformer/e2e_pipeline_5step_vis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["ped_crossing", "divider", "boundary"]


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


def sync():
    check_cuda(cudart.cudaDeviceSynchronize()[0], "cudaDeviceSynchronize failed")


def make_event():
    err, ev = cudart.cudaEventCreate()
    check_cuda(err, "cudaEventCreate failed")
    return ev


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
            mode = "IN " if engine.binding_is_input(i) else "OUT"
            dtype = engine.get_binding_dtype(i)
            shape = tuple(engine.get_binding_shape(i))
            print(f"  {mode} {i}: {bname} {shape} {dtype}")

    def bind_host_input(self, name, arr):
        idx = self.engine.get_binding_index(name)
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        arr = np.ascontiguousarray(arr.astype(dtype, copy=False))
        self.ctx.set_binding_shape(idx, arr.shape)

        ptr = cuda_malloc(arr.nbytes)
        memcpy_h2d(ptr, arr)

        self.owned_ptrs[f"input:{name}"] = ptr
        self.bindings[idx] = int(ptr)
        return ptr

    def bind_device_input(self, name, ptr, shape):
        idx = self.engine.get_binding_index(name)
        self.ctx.set_binding_shape(idx, tuple(shape))
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
                raise RuntimeError(f"{self.name}: unresolved output shape for {name}: {shape}")

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
            out[name] = host.copy()
        return out

    def output_ptr(self, name):
        return self.owned_ptrs[f"output:{name}"]

    def output_shape(self, name):
        return tuple(self.output_host[name].shape)

    def input_ptr(self, name):
        return self.owned_ptrs[f"input:{name}"]


def load_tensor_from_env(env_name, shape):
    path = os.environ.get(env_name, "")
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{env_name} points to missing file: {path}")

    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        arr = np.fromfile(path, dtype=np.float32).reshape(shape)

    if tuple(arr.shape) != tuple(shape):
        raise RuntimeError(f"{env_name} expected shape {shape}, got {arr.shape} from {path}")

    print(f"[inputs] loaded {env_name}: {path} {arr.shape} min/max {arr.min():.6f} {arr.max():.6f}")
    return arr


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

    # Optional explicit query input. This is the correct way to replay PyTorch's
    # torch.normal CUDA initial query. Do NOT fall back to parity_head.
    query_coords = _load_env_array("E2E_QUERY_COORDS", (1, 100, 20, 2), np.float32)

    if query_coords is None:
        seed = int(os.environ.get("E2E_SEED", "123"))
        rng = np.random.default_rng(seed)
        query_coords = rng.normal(0.5, 0.25, size=(1, 100, 20, 2)).astype(np.float32)
        query_coords = np.clip(query_coords, 0.0, 1.0)
        print("[inputs] using NumPy clipped-normal initial query_coords",
              query_coords.shape,
              float(query_coords.min()),
              float(query_coords.max()),
              float(query_coords.mean()))
    else:
        print("[inputs] using explicit E2E_QUERY_COORDS initial query",
              query_coords.shape,
              float(query_coords.min()),
              float(query_coords.max()),
              float(query_coords.mean()))

    return img, ego2img, query_coords

def get_timesteps():
    raw = os.environ.get("E2E_TIMESTEPS", "999,749,499,249,0")
    steps = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(steps) != 5:
        raise RuntimeError(f"E2E_TIMESTEPS must have 5 values, got {steps}")
    print("[sampling] timesteps:", steps)
    return steps


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def visualize(line_preds, cls_logits, out_png, score_thr=0.25, topk=50):
    """
    Visualize normalized line predictions.
    Assumes line_preds are normalized [0,1] and shaped [1,100,40].
    Converts to a simple BEV frame: x in [-30,30], y in [-15,15].
    """
    lines = line_preds.reshape(1, 100, 20, 2)[0]
    logits = cls_logits[0]
    probs = sigmoid(logits)

    labels = probs.argmax(axis=1)
    scores = probs.max(axis=1)

    order = np.argsort(-scores)
    keep = [i for i in order[:topk] if scores[i] >= score_thr]

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Draw ROI rectangle.
    ax.plot([-30, 30, 30, -30, -30], [-15, -15, 15, 15, -15], linewidth=1)

    for idx in keep:
        pts = lines[idx].copy()

        # normalized [0,1] -> metric BEV-ish coordinates
        x = pts[:, 0] * 60.0 - 30.0
        y = pts[:, 1] * 30.0 - 15.0

        label = int(labels[idx])
        score = float(scores[idx])
        name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)

        ax.plot(x, y, linewidth=1.5, label=f"{name} {score:.2f}" if idx == keep[0] else None)
        ax.scatter(x[0], y[0], s=8)

    ax.set_title(f"TRT MapDiffusion 5-step output | kept {len(keep)} / topk {topk}, score_thr {score_thr}")
    ax.set_xlabel("x meters")
    ax.set_ylabel("y meters")
    ax.set_xlim(-32, 32)
    ax.set_ylim(-17, 17)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Avoid huge legends.
    if keep:
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    print("[vis] saved:", out_png)
    print("[vis] kept:", len(keep))
    if keep:
        print("[vis] top scores/classes:")
        for idx in keep[:10]:
            label = int(labels[idx])
            name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
            print(f"  query={idx:03d} score={scores[idx]:.4f} class={name}")


def main():
    logger = trt.Logger(trt.Logger.WARNING)

    load_plugins()

    engine_a = load_engine(ENGINE_A, logger)
    engine_b = load_engine(ENGINE_B, logger)
    engine_d = load_engine(ENGINE_D, logger)
    engine_c = load_engine(ENGINE_C, logger)

    img, ego2img, query_coords = create_inputs()
    timesteps = get_timesteps()

    A = TrtModule("EngineA_BackboneFPN", engine_a)
    B = TrtModule("EngineB_BEVFormerEncoder", engine_b)
    D = TrtModule("EngineD_StreamFusionNeck", engine_d)
    C = TrtModule("EngineC_MapDiffusionHead", engine_c)

    # A: img -> feat0/feat1/feat2
    A.bind_host_input("img", img)
    A.allocate_outputs()

    # B: feat0/feat1/feat2 + ego2img -> bev_features
    B.bind_device_input("feat0", A.output_ptr("feat0"), A.output_shape("feat0"))
    B.bind_device_input("feat1", A.output_ptr("feat1"), A.output_shape("feat1"))
    B.bind_device_input("feat2", A.output_ptr("feat2"), A.output_shape("feat2"))
    B.bind_host_input("ego2img", ego2img)
    B.allocate_outputs()

    # D: raw BEV + raw BEV -> fused BEV
    D.bind_device_input("prev_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.bind_device_input("curr_bev", B.output_ptr("bev_features"), B.output_shape("bev_features"))
    D.allocate_outputs()

    # C: fused BEV + query_coords + timestep + prev_query_feat -> line_preds/cls_logits/query_feat
    C.bind_device_input("bev_features", D.output_ptr("fused_bev"), D.output_shape("fused_bev"))

    timestep_arr = np.array([timesteps[0]], dtype=np.float32)
    query_ptr = C.bind_host_input("query_coords", query_coords)
    timestep_ptr = C.bind_host_input("timestep", timestep_arr)

    # Temporal Route B: first frame starts with zero query memory.
    prev_query_feat = np.zeros((1, 100, 512), dtype=np.float32)
    prev_query_ptr = C.bind_host_input("prev_query_feat", prev_query_feat)

    C.allocate_outputs()

    def run_5step_once(copy_final=False):
        # A and B run once.
        A.execute()
        B.execute()
        D.execute()

        q = query_coords.copy()
        final_out = None
        prev_query_feat_local = np.zeros((1, 100, 512), dtype=np.float32)

        for step_idx, t in enumerate(timesteps):
            t_arr = np.array([t], dtype=np.float32)

            memcpy_h2d(query_ptr, q.astype(np.float32))
            memcpy_h2d(timestep_ptr, t_arr)
            memcpy_h2d(prev_query_ptr, prev_query_feat_local)

            C.execute()

            # Copy final and intermediate head output to CPU for feedback.
            # For latency, this includes D2H every step. Later we can move the
            # q-update to GPU if needed.
            out = C.copy_outputs_to_host()
            if "query_feat" in out:
                prev_query_feat_local = np.ascontiguousarray(out["query_feat"].astype(np.float32))
            line_preds = out["line_preds"]

            # Practical feedback loop: final normalized line prediction becomes next query.
            q = np.clip(line_preds.reshape(1, 100, 20, 2).astype(np.float32), 0.0, 1.0)

            if copy_final or step_idx == len(timesteps) - 1:
                final_out = out

        return final_out

    print("\n============================================================")
    print("Smoke 5-step run")
    print("============================================================")
    final = run_5step_once(copy_final=True)
    sync()

    line_preds = final["line_preds"]
    cls_logits = final["cls_logits"]

    np.savez(
        OUT_DIR / "trt_5step_outputs.npz",
        line_preds=line_preds,
        cls_logits=cls_logits,
        timesteps=np.asarray(timesteps, dtype=np.float32),
    )

    print("line_preds:", line_preds.shape, line_preds.dtype, float(line_preds.min()), float(line_preds.max()), float(line_preds.mean()))
    print("cls_logits:", cls_logits.shape, cls_logits.dtype, float(cls_logits.min()), float(cls_logits.max()), float(cls_logits.mean()))

    score_thr = float(os.environ.get("E2E_SCORE_THR", "0.25"))
    topk = int(os.environ.get("E2E_TOPK", "50"))

    visualize(
        line_preds,
        cls_logits,
        OUT_DIR / "trt_5step_bev.png",
        score_thr=score_thr,
        topk=topk,
    )

    print("\n============================================================")
    print("Latency: A + B + 5 x C feedback loop")
    print("============================================================")

    warmup = int(os.environ.get("E2E_WARMUP", "10"))
    iters = int(os.environ.get("E2E_ITERS", "50"))

    for _ in range(warmup):
        run_5step_once(copy_final=False)
    sync()

    start = make_event()
    end = make_event()

    check_cuda(cudart.cudaEventRecord(start, 0), "record start failed")
    for _ in range(iters):
        run_5step_once(copy_final=False)
    check_cuda(cudart.cudaEventRecord(end, 0), "record end failed")
    check_cuda(cudart.cudaEventSynchronize(end), "event sync failed")

    err, total_ms = cudart.cudaEventElapsedTime(start, end)
    check_cuda(err, "elapsed failed")

    avg_ms = total_ms / iters
    print(f"Warmup: {warmup}")
    print(f"Iters: {iters}")
    print(f"Average latency A+B+5C feedback: {avg_ms:.3f} ms")
    print("Note: includes CPU D2H/H2D feedback between diffusion steps.")
    print("Saved:", OUT_DIR)


if __name__ == "__main__":
    main()
